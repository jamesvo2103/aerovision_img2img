import os
import gc
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import math

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats

from pix2pix_turbo import Pix2Pix_Turbo
from my_utils.training_utils import parse_args_paired_training, PairedDataset

def sobel_filter(images):
    """
    Applies a Sobel filter to a batch of images to detect edges.
    Args:
        images (torch.Tensor): A batch of images of shape (B, C, H, W).
    Returns:
        torch.Tensor: The magnitude of the gradients.
    """
    # Convert to grayscale for gradient calculation
    if images.shape[1] > 1:
        images_gray = transforms.Grayscale()(images)
    else:
        images_gray = images

    # Define Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=images.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=images.device).view(1, 1, 3, 3)

    # Apply filters
    grad_x = F.conv2d(images_gray, sobel_x, padding=1)
    grad_y = F.conv2d(images_gray, sobel_y, padding=1)

    # Calculate gradient magnitude
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    return grad_magnitude
def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    if args.pretrained_model_name_or_path == "stabilityai/sd-turbo":
        net_pix2pix = Pix2Pix_Turbo(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae)
        net_pix2pix.set_train()

    if args.resume_from_checkpoint is not None:
        print(f"Loading weights from checkpoint: {args.resume_from_checkpoint}")
    
    # Load the custom checkpoint dictionary
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
    
    # Create a new, empty dictionary to hold the correctly formatted weights
        reconstructed_state_dict = {}
    
    # Re-prefix the keys from the saved UNet state dict
        if "state_dict_unet" in checkpoint:
            for key, value in checkpoint["state_dict_unet"].items():
                reconstructed_state_dict[f"unet.{key}"] = value
    
    # Re-prefix the keys from the saved VAE state dict
        if "state_dict_vae" in checkpoint:
            for key, value in checkpoint["state_dict_vae"].items():
                reconstructed_state_dict[f"vae.{key}"] = value
    
    # Load the weights. `strict=False` is crucial because the checkpoint
    # only contains the trained LoRA layers. The base model weights are
    # already loaded, and we are just loading our fine-tuned changes on top.
        net_pix2pix.load_state_dict(reconstructed_state_dict, strict=False)
    
    print("Weights successfully loaded from checkpoint.")
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_clip, _ = clip.load("ViT-B/32", device="cuda")
    net_clip.requires_grad_(False)
    net_clip.eval()

    net_lpips.requires_grad_(False)

    # make the optimizer
    layers_to_opt = []
    for n, _p in net_pix2pix.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_pix2pix.unet.conv_in.parameters())
    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt = layers_to_opt + list(net_pix2pix.vae.decoder.skip_conv_1.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_2.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_3.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.gan_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    disc_lr = args.learning_rate if args.disc_learning_rate is None else args.disc_learning_rate
    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=disc_lr,
    betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
            num_warmup_steps=args.gan_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    dataset_train = PairedDataset(dataset_folder=args.dataset_folder, image_prep=args.train_image_prep, split="train", tokenizer=net_pix2pix.tokenizer)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dataset_val = PairedDataset(dataset_folder=args.dataset_folder, image_prep=args.test_image_prep, split="test", tokenizer=net_pix2pix.tokenizer)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Prepare everything with our `accelerator`.
    net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    net_clip, net_lpips = accelerator.prepare(net_clip, net_lpips)
    # renorm with image net statistics
    t_clip_renorm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_pix2pix.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_clip.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # turn off eff. attn for the discriminator
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # compute the reference stats for FID tracking
    if accelerator.is_main_process and args.track_val_fid:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)

        def fn_transform(x):
            x_pil = Image.fromarray(x)
            out_pil = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS)(x_pil)
            return np.array(out_pil)

        ref_stats = get_folder_features(os.path.join(args.dataset_folder, "test_B"), model=feat_model, num_workers=0, num=None,
                shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)

    

    global_step = 0
    best_val_loss = float("inf")
    lossG = torch.tensor(0.0)
    lossD = torch.tensor(0.0)
    loss_clipsim = torch.tensor(0.0)

    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            # --- Define variables in the correct scope ---
            x_src = batch["conditioning_pixel_values"]
            x_tgt = batch["output_pixel_values"]
            B, C, H, W = x_src.shape
             

            with accelerator.accumulate(net_pix2pix, net_disc):
                # =================================================================================
                #                                 Generator Update
                # =================================================================================

                # --- 1. Cosine Annealing for GAN weight ---
                gan_weight = 0.0
                warmup_end_step = args.gan_initial_warmup_steps
                rampup_end_step = args.gan_initial_warmup_steps + args.gan_ramp_up_steps
                if global_step > rampup_end_step:
    # After ramp-up, use full weight
                    gan_weight = 1.0
                elif global_step > warmup_end_step:
    # During ramp-up, linearly increase weight from 0 to 1
                    progress = (global_step - warmup_end_step) / float(args.gan_ramp_up_steps)
                    gan_weight = progress
                else:
    # During initial warm-up, weight is 0
                    gan_weight = 0.0

                current_lambda_gan = args.lambda_gan * gan_weight

                # --- 2. Calculate combined generator loss ---
                x_tgt_pred = net_pix2pix(x_src, prompt_tokens=batch["input_ids"], deterministic=True)
                airfoil_mask = batch["airfoil_mask"]
                
                loss_l2 = F.mse_loss(x_tgt_pred * airfoil_mask, x_tgt.float() * airfoil_mask) * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred * airfoil_mask, x_tgt.float() * airfoil_mask).mean() * args.lambda_lpips
                
                total_gen_loss = loss_l2 + loss_lpips

                if args.lambda_gradient > 0:
                    # Calculate gradients of the predicted and target images
                    pred_grads = sobel_filter(x_tgt_pred)
                    target_grads = sobel_filter(x_tgt)
                    
                    # Calculate the L1 loss between the gradients, applying the mask
                    loss_gradient = F.l1_loss(pred_grads * airfoil_mask, target_grads * airfoil_mask) * args.lambda_gradient
                    total_gen_loss += loss_gradient
                    
                if args.lambda_clipsim > 0:
                    x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                    x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                    caption_tokens = clip.tokenize(batch["caption"], truncate=True).to(x_tgt_pred.device)
                    clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                    loss_clipsim = (1 - clipsim.mean() / 100)
                    total_gen_loss += loss_clipsim * args.lambda_clipsim
                
                lossG = net_disc(x_tgt_pred, for_G=True).mean() * current_lambda_gan
                total_gen_loss += lossG
                
                # --- 3. Perform a single update for the Generator ---
                accelerator.backward(total_gen_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # =================================================================================
                #                                 Discriminator Update
                # =================================================================================
                if current_lambda_gan > 0:
                    lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * current_lambda_gan
                    accelerator.backward(lossD_real)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
    
                    lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * current_lambda_gan
                    accelerator.backward(lossD_fake)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    lossD = lossD_real + lossD_fake
            
            # --- Logging, checkpointing, and validation ---
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    logs = {}
                    logs["lossG"] = lossG.detach().item()
                    logs["lossD"] = lossD.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    if args.lambda_clipsim > 0:
                        logs["loss_clipsim"] = loss_clipsim.detach().item()
                    progress_bar.set_postfix(**logs)


                    # viz some images
                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/source": [wandb.Image(x_src[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image(x_tgt[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_output": [wandb.Image(x_tgt_pred[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)

                    # compute validation set FID, L2, LPIPS, CLIP-SIM
                    if global_step % args.eval_freq == 1:
                        l_l2, l_lpips, l_clipsim = [], [], []
                        if args.track_val_fid:
                            os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt = batch_val["output_pixel_values"].cuda()
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                x_tgt_pred = accelerator.unwrap_model(net_pix2pix)(x_src, prompt_tokens=batch_val["input_ids"].cuda(), deterministic=True)
                                # compute the reconstruction losses
                                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
                                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean()
                                # compute clip similarity loss
                                x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                                x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                                caption_tokens = clip.tokenize(batch_val["caption"], truncate=True).to(x_tgt_pred.device)
                                clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                                clipsim = clipsim.mean()

                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())
                                l_clipsim.append(clipsim.item())
                            # save output images to file for FID evaluation
                            if args.track_val_fid:
                                output_pil = transforms.ToPILImage()(x_tgt_pred[0].cpu() * 0.5 + 0.5)
                                outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"val_{step}.png")
                                output_pil.save(outf)
                        if args.track_val_fid:
                            curr_stats = get_folder_features(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), model=feat_model, num_workers=0, num=None,
                                    shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                                    mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)
                            fid_score = fid_from_feats(ref_stats, curr_stats)
                            logs["val/clean_fid"] = fid_score
                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        logs["val/clipsim"] = np.mean(l_clipsim)
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
