"""
Phase 2 training — Johnson et al. feed-forward style transfer.

Quick-start:
    python train.py \
        --coco_root /data/coco \
        --style_img  style/picasso.jpg \
        --output_dir checkpoints/picasso \
        --epochs 2

Overfit sanity check (recommended before full training):
    python train.py \
        --coco_root /data/coco \
        --style_img  style/picasso.jpg \
        --output_dir checkpoints/debug \
        --epochs 1 \
        --debug_batches 8
"""

import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from PIL import Image
import torchvision.transforms as T

from model   import TransformNet
from loss    import PerceptualLoss
from dataset import make_dataloader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path: str, device: torch.device) -> torch.Tensor:
    """Load a single image as a (1, 3, H, W) tensor in [0, 1]."""
    img = Image.open(path).convert('RGB')
    return T.ToTensor()(img).unsqueeze(0).to(device)


def save_checkpoint(model, optimizer, epoch, step, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch':           epoch,
        'step':            step,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, path)
    print(f'  Saved checkpoint → {path}')


def load_checkpoint(path: str, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    print(f'Resumed from {path} (epoch {ckpt["epoch"]}, step {ckpt["step"]})')
    return ckpt['epoch'], ckpt['step']


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    # --- Model ---
    model = TransformNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch, global_step = 0, 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, device
        )

    # --- Style image & loss ---
    style_img = load_image(args.style_img, device)
    loss_fn   = PerceptualLoss(
        device, style_img,
        alpha=args.alpha,
        beta=args.beta,
    )

    # --- Data ---
    loader = make_dataloader(
        args.coco_root,
        split=args.split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Tip: run with --debug_batches 8 first.
    # Loss should fall visibly within ~50 steps. If it doesn't, the loss
    # computation has a bug — check VGG normalisation and style gram shapes.
    # ---------------------------------------------------------------------------

    print(f'\nStarting training for {args.epochs} epoch(s).')
    print(f'Style image   : {args.style_img}')
    print(f'alpha={args.alpha}, beta={args.beta:.0e}\n')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, content in enumerate(loader):

            # Debug mode: stop after N batches
            if args.debug_batches and batch_idx >= args.debug_batches:
                break

            content = content.to(device)

            # Forward
            generated = model(content)

            # Loss
            loss, l_c, l_s = loss_fn(generated, content)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f'epoch {epoch+1}/{args.epochs} | '
                    f'step {global_step:6d} | '
                    f'loss {loss.item():.4f} | '
                    f'content {l_c:.4f} | '
                    f'style {l_s:.6f} | '
                    f'{elapsed:.1f}s'
                )
                t0 = time.time()

            if global_step % args.save_every == 0:
                save_checkpoint(
                    model, optimizer, epoch, global_step,
                    out_dir / f'step_{global_step:06d}.pt'
                )

        avg = epoch_loss / (batch_idx + 1)
        print(f'\nEpoch {epoch+1} done — avg loss: {avg:.4f}')
        save_checkpoint(
            model, optimizer, epoch + 1, global_step,
            out_dir / f'epoch_{epoch+1:02d}.pt'
        )

    # Save final model weights only (small file, easy to share)
    final_path = out_dir / 'model_final.pt'
    torch.save(model.state_dict(), final_path)
    print(f'\nTraining complete. Final weights → {final_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # Paths
    p.add_argument('--coco_root',   required=True,
                   help='Root folder of COCO images')
    p.add_argument('--style_img',   required=True,
                   help='Path to the style image (e.g. picasso.jpg)')
    p.add_argument('--output_dir',  required=True,
                   help='Where to save checkpoints')
    p.add_argument('--resume',      default=None,
                   help='Path to a checkpoint to resume from')
    p.add_argument('--split',       default='train2017',
                   help='Subfolder inside coco_root (default: train2017)')

    # Training
    p.add_argument('--epochs',      type=int,   default=2)
    p.add_argument('--batch_size',  type=int,   default=8)
    p.add_argument('--img_size',    type=int,   default=256)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--num_workers', type=int,   default=4)

    # Loss weights
    p.add_argument('--alpha', type=float, default=1.0,
                   help='Content loss weight')
    p.add_argument('--beta',  type=float, default=1e6,
                   help='Style loss weight')

    # Logging
    p.add_argument('--log_every',  type=int, default=100,
                   help='Print loss every N steps')
    p.add_argument('--save_every', type=int, default=2000,
                   help='Save checkpoint every N steps')

    # Debug
    p.add_argument('--debug_batches', type=int, default=0,
                   help='If > 0, stop each epoch after this many batches '
                        '(overfitting sanity check)')

    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())