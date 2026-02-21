import argparse
from pathlib import Path
from typing import TypedDict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from u_net.u_net import UNetLightning
from u_net.datasets.pet_dataset import PetDatasetLightning


class Arguments(TypedDict):
    seed: int
    data_path: str
    ann_path: str
    batch_size: int
    num_workers: int
    device: str
    num_classes: int
    gray_scale: bool
    model_save_path: str
    epochs: int
    accelerator: str
    lr: float


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ann_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--gray_scale", action="store_true")
    parser.add_argument("--model_save_path", type=str, default="model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=0.001)

    return Arguments(**vars(parser.parse_args()))


def main() -> None:
    args = parse_args()
    pl.seed_everything(args["seed"])

    datamodule = PetDatasetLightning(
        data_path=args["data_path"],
        annotations_path=args["ann_path"],
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        seed=args["seed"],
    )
    datamodule.setup()

    model = UNetLightning(
        channels=1 if args["gray_scale"] else 3,
        num_outputs=args["num_classes"],
        lr=args["lr"]
    )

    model_save_dir = Path(args["model_save_path"]).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        model_save_dir,
        filename="model",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = TensorBoardLogger(save_dir=args["model_save_path"], name="model_logs")

    trainer = pl.Trainer(
        max_epochs=args["epochs"],
        accelerator=args["accelerator"],
        devices=args["device"],
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        logger=logger,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
