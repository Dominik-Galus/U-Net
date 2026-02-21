from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PetDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        annotations_path: Path,
        img_transforms: transforms.Compose | None = None,
        ann_transforms: transforms.Compose | None = None,
    ) -> None:
        self.data_path = data_path
        self.annotations_path = annotations_path
        self.img_transforms = img_transforms if img_transforms else transforms.Compose(
            [
                transforms.Resize(size=(576, 576)),
                transforms.ToTensor()
            ]
        )

        self.ann_transforms = ann_transforms if ann_transforms else transforms.Resize(
            size=(576, 576),
            interpolation=transforms.InterpolationMode.NEAREST
        )

        data = sorted([img for img in data_path.glob("*.jpg")])
        annotations = sorted([ann for ann in self.annotations_path.glob("*.png")])

        self.data = [(img, ann) for img, ann in zip(data, annotations)]

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, ann_path = self.data[index]
        img = Image.open(img_path).convert("RGB")
        ann_img = Image.open(ann_path).convert("P")
        ann = torch.from_numpy(np.array(ann_img)).long() - 1
        img_tensor = self.img_transforms(img)
        ann = self.ann_transforms(ann.unsqueeze(dim=0)).squeeze()
        return img_tensor, ann
