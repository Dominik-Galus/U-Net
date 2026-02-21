import pytorch_lightning as pl
import torch
from torch import isin, nn
from torch.nn import functional as f
from torchmetrics import Accuracy
from torchvision.transforms import functional

from u_net.building_blocks import DownBlock, UpBlock


class UNet(nn.Module):
    def __init__(self, channels: int, num_outputs: int) -> None:
        super().__init__()
        self.down_blocks = nn.ModuleList([
            DownBlock(channels, 64),
            DownBlock(64, 128),
            DownBlock(128, 256),
            DownBlock(256, 512),
        ])
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3),
            nn.ReLU(),
        )
        self.len_dblocks = len(self.down_blocks)

        self.up_blocks = nn.ModuleList([
            UpBlock(in_channels=1024, out_channels=512),
            UpBlock(in_channels=512, out_channels=256),
            UpBlock(in_channels=256, out_channels=128),
            UpBlock(in_channels=128, out_channels=64),
        ])

        self.out_conv = nn.Conv2d(
            in_channels=64, out_channels=num_outputs, kernel_size=1
        )

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dblocks_result = []
        for block in self.down_blocks:
            x, to_copy = block(x)
            dblocks_result.append(to_copy)
        x = self.down_conv(x)

        for i, block in enumerate(self.up_blocks):
            x = block(x, dblocks_result[self.len_dblocks - i - 1])
        x = self.out_conv(x)
        return x

    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(
                m.weight,
                mode="fan_out",
                nonlinearity="relu"
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class UNetLightning(pl.LightningModule):
    def __init__(self, channels: int, num_outputs: int, lr: float = 0.001) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(channels=channels, num_outputs=num_outputs)
        self.lr = lr

        self.train_acc = Accuracy("multiclass", num_classes=num_outputs)
        self.val_acc = Accuracy("multiclass", num_classes=num_outputs)
        self.test_acc = Accuracy("multiclass", num_classes=num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def common_step(self, img: torch.Tensor, ann: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.model(img)
        ann = functional.center_crop(ann, output_size=(logits.shape[-2], logits.shape[-1]))
        ann = ann.squeeze(1).long()
        loss = f.cross_entropy(logits, ann)
        return logits, loss, ann

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        img, ann = batch
        logits, loss, ann = self.common_step(img, ann)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, ann)

        self.log("train/loss", loss, on_epoch=True)
        self.log("train/acc", self.train_acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        img, ann = batch
        logits, loss, ann = self.common_step(img, ann)
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, ann)

        self.log("val/loss", loss, on_epoch=True)
        self.log("val/acc", self.val_acc, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        img, ann = batch
        logits, loss, ann = self.common_step(img, ann)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, ann)

        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", self.test_acc, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
