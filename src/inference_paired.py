import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pix2pix_turbo import Pix2Pix_Turbo
from image_prep import canny_from_pil
import cv2 # Import OpenCV

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='path to the input image')
    parser.add_argument('--prompt', type=str, required=True, help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--low_threshold', type=int, default=100, help='Canny low threshold')
    parser.add_argument('--high_threshold', type=int, default=200, help='Canny high threshold')
    parser.add_argument('--gamma', type=float, default=0.4, help='The sketch interpolation guidance amount')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name == '' and args.model_path == '':
        raise ValueError('Either model_name or model_path should be provided')

    os.makedirs(args.output_dir, exist_ok=True)

    # initialize the model
    model = Pix2Pix_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.set_eval()
    if args.use_fp16:
        model.half()

    # make sure that the input image is a multiple of 8
    input_image = Image.open(args.input_image).convert('RGB')
    new_width = input_image.width - input_image.width % 8
    new_height = input_image.height - input_image.height % 8
    input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
    bname = os.path.basename(args.input_image)

    # translate the image
    with torch.no_grad():
        if args.model_name == 'edge_to_image':
            # This part is for the original pretrained models, not your fine-tuned one
            canny = canny_from_pil(input_image, args.low_threshold, args.high_threshold)
            canny_viz_inv = Image.fromarray(255 - np.array(canny))
            canny_viz_inv.save(os.path.join(args.output_dir, bname.replace('.png', '_canny.png')))
            c_t = F.to_tensor(canny).unsqueeze(0).cuda()
            if args.use_fp16:
                c_t = c_t.half()
            output_image = model(c_t, args.prompt)

        elif args.model_name == 'sketch_to_image_stochastic':
            # This part is for the original pretrained models, not your fine-tuned one
            image_t = F.to_tensor(input_image) < 0.5
            c_t = image_t.unsqueeze(0).cuda().float()
            torch.manual_seed(args.seed)
            B, C, H, W = c_t.shape
            noise = torch.randn((1, 4, H // 8, W // 8), device=c_t.device)
            if args.use_fp16:
                c_t = c_t.half()
                noise = noise.half()
            output_image = model(c_t, args.prompt, deterministic=False, r=args.gamma, noise_map=noise)

        else:
            # --- THIS IS THE CORRECTED LOGIC FOR YOUR FINE-TUNED MODEL ---
            # This is the path your model will use when model_name is not specified
            
            # 1. Convert the 3-channel input image to a tensor
            input_t = F.to_tensor(input_image).unsqueeze(0).cuda()

            # 2. Create an empty fourth channel (all zeros)
            _, _, H, W = input_t.shape
            empty_channel = torch.zeros(1, 1, H, W, device=input_t.device, dtype=input_t.dtype)
            
            # 3. Combine them to create the 4-channel input the model expects
            c_t = torch.cat([input_t, empty_channel], dim=1)

            if args.use_fp16:
                c_t = c_t.half()
            
            # 4. Run inference
            output_image = model(c_t, args.prompt)

        # Convert the raw model output (flow only) to a PIL Image
        flow_only_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)

    # --- COMPOSITING STEP (No changes needed here) ---
    input_cv = cv2.imread(args.input_image)
    flow_cv = cv2.cvtColor(np.array(flow_only_pil), cv2.COLOR_RGB2BGR)

    h, w, _ = input_cv.shape
    flow_cv = cv2.resize(flow_cv, (w, h))

    gray_flow = cv2.cvtColor(flow_cv, cv2.COLOR_BGR2GRAY)
    mask = gray_flow > 0

    final_image = input_cv.copy()
    final_image[mask] = flow_cv[mask]

    composite_filename = bname.replace('.png', '_composite.png')
    cv2.imwrite(os.path.join(args.output_dir, composite_filename), final_image)
    
    print(f"✅ Final composite image saved to: {os.path.join(args.output_dir, composite_filename)}")
