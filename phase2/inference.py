"""
Run a trained transform network on one or more content images.

Single image:
    python inference.py \
        --weights checkpoints/picasso/model_final.pt \
        --input   photos/duck.jpg \
        --output  results/duck_picasso.jpg

Batch (whole folder):
    python inference.py \
        --weights checkpoints/picasso/model_final.pt \
        --input   photos/ \
        --output  results/
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

from model import TransformNet


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def load_model(weights_path: str, device: torch.device) -> TransformNet:
    model = TransformNet().to(device)
    state = torch.load(weights_path, map_location=device)
    # Handle both full checkpoints and plain state dicts
    if 'model_state' in state:
        state = state['model_state']
    model.load_state_dict(state)
    model.eval()
    print(f'Loaded weights from {weights_path}')
    return model


def stylize(model, img_path: Path, device: torch.device, img_size: int = None) -> Image.Image:
    """
    Stylize a single image.
    If img_size is None, the image is processed at its original resolution.
    """
    img = Image.open(img_path).convert('RGB')
    original_size = img.size  # (W, H) — we'll restore this

    transforms = []
    if img_size:
        transforms += [T.Resize(img_size), T.CenterCrop(img_size)]
    transforms.append(T.ToTensor())
    to_tensor = T.Compose(transforms)

    tensor = to_tensor(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    with torch.no_grad():
        output = model(tensor)  # (1, 3, H, W) in [0, 1]

    # Convert back to PIL
    out_img = T.ToPILImage()(output.squeeze(0).cpu())

    # Restore original size if we didn't resize
    if not img_size:
        out_img = out_img.resize(original_size, Image.LANCZOS)

    return out_img


def run(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    model    = load_model(args.weights, device)
    in_path  = Path(args.input)
    out_path = Path(args.output)

    # Collect input files
    if in_path.is_dir():
        files = sorted([p for p in in_path.iterdir() if p.suffix.lower() in IMG_EXTS])
        out_path.mkdir(parents=True, exist_ok=True)
        is_batch = True
    else:
        files = [in_path]
        is_batch = False

    if not files:
        print(f'No images found at {in_path}')
        return

    for img_path in files:
        result = stylize(model, img_path, device, img_size=args.img_size)

        if is_batch:
            save_to = out_path / img_path.name
        else:
            save_to = out_path
            save_to.parent.mkdir(parents=True, exist_ok=True)

        result.save(save_to)
        print(f'  {img_path.name} → {save_to}')

    print(f'\nDone. {len(files)} image(s) stylized.')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights',  required=True, help='Path to model_final.pt or checkpoint')
    p.add_argument('--input',    required=True, help='Input image or folder')
    p.add_argument('--output',   required=True, help='Output image path or folder')
    p.add_argument('--img_size', type=int, default=None,
                   help='Resize input before stylizing (default: use original size)')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())