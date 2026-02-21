# U-Net
![Architecture](resources/u-net-architecture.png)

My architecture implementation from paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## Install
using package manager uv:
```shell
uv venv && uv sync
```
using python venv:
```shell
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install .
```

### Training
You can train the model using script:
```shell
python3 train.py [here arguments for script defined in script]
```

## Todo
- Add option to train with Dice Loss
