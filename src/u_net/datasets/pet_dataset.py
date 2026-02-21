from pathlib import Path

import pytorch_lightning as pl
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
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


class PetDatasetLightning(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        annotations_path: str,
        img_transforms: transforms.Compose | None = None,
        ann_transforms: transforms.Compose | None = None,
        val_split: float = 0.15,
        batch_size: int = 1,
        num_workers: int = 1,
        seed: int = 67,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.ann_path = annotations_path
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
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        root_data_path = Path(self.data_path)
        if (root_data_path / "train").exists() and (root_data_path / "test").exists():
            train_data_path = root_data_path / "train"
            test_data_path = root_data_path / "test"
        else:
            train_data_path = root_data_path
            test_data_path = root_data_path

        root_ann_path = Path(self.ann_path)
        if (root_ann_path / "train").exists() and (root_ann_path / "test").exists():
            train_ann_path = root_ann_path / "train"
            test_ann_path = root_ann_path / "test"
        else:
            train_ann_path = root_ann_path
            test_ann_path = root_ann_path

        self.test_ds = PetDataset(
            data_path=test_data_path,
            annotations_path=test_ann_path,
            img_transforms=self.img_transforms,
            ann_transforms=self.ann_transforms,
        )

        dataset = PetDataset(
            data_path=train_data_path,
            annotations_path=train_ann_path,
            img_transforms=self.img_transforms,
            ann_transforms=self.ann_transforms,
        )
        n_val = int(len(dataset) * self.val_split)
        n_train = len(dataset) - n_val
        self.train_ds, self.val_ds = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
