# SPCNet

## Introduction

Serial Pyramid Convolutional Network (SPCNet) is a deep convolutional network designed for remote sensing object detection tasks. Our network employs serial small-kernel convolutions to achieve multi-scale feature extraction, effectively maintaining receptive field coverage while reducing computational complexity. In this repository, the model is referred to as MSCNet to match the pre-trained weights. This documentation provides detailed instructions for installation, training, and testing procedures, along with locations of model weights and related configuration files.

## Installation

```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install yapf==0.40.1

pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

pip install -U openmim
mim install mmdet
mim install mmengine
git clone 
cd SPCNet
mim install -v -e .
cd mmpretrain
pip install -v -e .
cd ..
```

If you encounter version mismatch issues with mim installation, you may download mmdetection-2.28.2 and mmengine-0.10.4 offline from the following URLs:

[mmdetection-2.28.2](https://github.com/open-mmlab/mmdetection/releases/tag/v2.28.2)

[mmengine-0.10.4](https://github.com/open-mmlab/mmengine/releases/tag/v0.10.4)

## Model

| Model | Checkpoint | Config |
|-----------|-----------|-----------|
| MSCNet    | [checkpoint/latest.pth]( https://pan.baidu.com/s/1yxVU83ZGqz6LRRUXa8ASZw?pwd=na3f)   | [configs/mscnet/mscnet-s_fpn_o-rcnn-dotav1-ms_le90.py](configs/mscnet/mscnet-s_fpn_o-rcnn-dotav1-ss_le90.py)    |

## Pre-trained Models

1. Download the ImageNet-1K dataset

ImageNet dataset download link: [ImageNet](https://image-net.org/download.php)

Please save the dataset in the mmpretrain/data folder and name it imagenet.

2. Pre-training

```
cd mmpretrain

# Single GPU Pre-training
python tools/train.py configs/mscnet/mscnet_8xb32_in1k.py --work-dir work_dirs/mscnet_8xb32_in1k

# Multi-GPU Pre-training
chmod +x ./tools/dist_train.sh
./tools/dist_train.sh configs/mscnet/mscnet_8xb32_in1k.py ${GPU_NUM}
```

## Training

1. Download the DOTA-v1.0 dataset:

DOTA-v1.0 dataset download link: [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html)

Please save the dataset in the data folder and name it DOTA.

2. Dataset Cropping

```

cd ..

python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_trainval.json

python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_val.json

python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_test.json

python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ms_trainval.json

python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ms_val.json

python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ms_test.json
```

3. Training

```
# single-scale

# Single GPU Training
python tools/train.py configs/mscnet/mscnet-s_fpn_o-rcnn-dotav1-ss_le90.py --work-dir work_dirs/mscnet-s_fpn_o-rcnn-dotav1-ss_le90

# Multi-GPU Training
./tools/dist_train.sh configs/mscnet/mscnet-s_fpn_o-rcnn-dotav1-ss_le90.py 8

# mmulti-scale

CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mscnet/mscnet-s_fpn_o-rcnn-dotav1-ms_le90.py --work-dir work_dirs/mscnet-s_fpn_o-rcnn-dotav1-ms_le90_1

./tools/dist_train.sh configs/mscnet/mscnet-s_fpn_o-rcnn-dotav1-ms_le90.py 8
```

## Test

```
# Single GPU Test
python tools/test.py configs/mscnet/mscnet-s_fpn_o-rcnn-dotav1-ms_le90.py checkpoint/mAP_best_epoch_60.pth --format-only

# Multi-GPU Test
./tools/dist_test.sh \
configs/mscnet/mscnet-s_fpn_o-rcnn-dotav1-ms_le90.py \
checkpoint/mAP_best_epoch_60.pth \
${GPU_NUM} \
--format-only
```
