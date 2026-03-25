import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class COCODataset(Dataset):
    """
    Flat image folder dataset — works with COCO or any folder of JPEGs/PNGs.

    Expected layout (after downloading COCO 2017 train):
        coco_root/
            train2017/
                000000000009.jpg
                000000000025.jpg
                ...

    Pass coco_root = '/path/to/coco', split = 'train2017'.

    If you just want to test quickly, point it at any folder of images and
    set split = '' (empty string) so it reads coco_root directly.
    """

    def __init__(
        self,
        coco_root: str,
        split:     str = 'train2017',
        img_size:  int = 256,
    ):
        root = Path(coco_root) / split if split else Path(coco_root)
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.paths = sorted([
            p for p in root.iterdir()
            if p.suffix.lower() in exts
        ])
        if not self.paths:
            raise FileNotFoundError(f'No images found in {root}')

        self.transform = T.Compose([
            T.Resize(img_size),               # shorter side → img_size
            T.CenterCrop(img_size),           # square crop
            T.ToTensor(),                     # [0, 255] → [0.0, 1.0]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)


def make_dataloader(
    coco_root: str,
    split:     str = 'train2017',
    img_size:  int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
) -> DataLoader:
    dataset = COCODataset(coco_root, split, img_size)
    print(f'Dataset: {len(dataset):,} images from {coco_root}/{split}')
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,    # keep batch sizes uniform
    )