
# GSNet for Object Detection in Aerial Images

This repository contains the official PyTorch implementation of the following paper:

**Learning to Reduce Information Bottleneck for Object Detection in Aerial Images**

Yuchen Shen, Dong Zhang, Zhihao Song, Xuesong Jiang, Qiaolin Ye

https://arxiv.org/abs/2204.02033v2

## Abstract
<p align="justify">
Object detection in aerial images is a critical and essential task in the fields of geoscience and remote sensing. Despite the popularity of computer vision methods in detecting objects in aerial images, these methods have been faced with significant limitations such as appearance occlusion and variable image sizes. In this letter, we explore the limitations of conventional neck networks in object detection by analyzing information bottlenecks. We propose an enhanced neck network to address the information deficiency issue in current neck networks. Our proposed neck network serves as a bridge between the backbone network and the head network. The enhanced neck network comprises a global semantic network (GSNet) and a feature fusion refinement module (FRM). The GSNet is designed to perceive contextual surroundings and propagate discriminative knowledge through a bidirectional global pattern. The FRM is developed to exploit different levels of features to capture comprehensive location information. We validate the efficacy and efficiency of our approach through experiments conducted on two challenging datasets, DOTA and HRSC2016. Our method outperforms existing approaches in terms of accuracy and complexity, demonstrating the superiority of our proposed method. The code has been open-sourced at GSNet.

## Introduction
This repository is created to build benchmarks for object detection in aerial images. The master branch works with **PyTorch 1.1** or higher. If you would like to use PyTorch 0.4.1, please checkout to the [PyTorch-0.4.1](https://github.com/open-mmlab/mmdetection/tree/pytorch-0.4.1) branch.

## Results
![Baseline and GSNet results](/show/fig3.png)
 Comparisons with the baseline on DOTA for oriented object detection with ResNet-101.

## Usage

#### Download imagenet pre-trained weights:
- [ResNet50](https://drive.google.com/file/d/1mQ9S0FzFpPHnocktH0DGVysufGt4tH0M/view?usp=sharing)
- [ResNet101](https://drive.google.com/file/d/1qlVf58T0fY4dddKst5i7-CL3DXhBi3Mp/view?usp=sharing)

#### Installation
Please refer to [INSTALL.md](INSTALL.md) for installation.

    
#### Get started
Please see [GETTING_STARTED.md](GETTING_STARTED.md) for the basic usage of mmdetection.

## Our weights
#### Our results on the test of DOTA.
|Backbone|Detector|+GS|Weight|mAP(%)|
|:---:|:---:|:---:|:---:|:---:|
|ResNet-101|Faster R-CNN||[download](https://github.com/ssyc123/GSNet/releases/download/v1.0/FastRCNN_DOTA_Baseline.pth)|73.09|
|ResNet-101|Faster R-CNN|+|[download](https://github.com/ssyc123/GSNet/releases/download/v1.0/FastRCNN_DOTA_GSNet.pth)|79.37|

#### Our results on HRSC2016.
|Backbone|Detector|Weight|mAP(%)|
|:---:|:---:|:---:|:---:|
|ResNet-101|Faster R-CNN|[download](https://github.com/ssyc123/GSNet/releases/download/v1.0/FastRcnn_HRSC_GSNet.pth)|90.50|

## Citing
If you use our work, please consider citing:
```
@article{shen2023learning,
  title={Learning to reduce information bottleneck for object detection in aerial images},
  author={Shen, Yuchen and Zhang, Dong and Song, Zhihao and Jiang, Xuesong and Ye, Qiaolin},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgements
[PyTorch](https://pytorch.org/), [MMDetection](https://github.com/open-mmlab/mmdetection), [AerialDetection](https://github.com/dingjiansw101/AerialDetection)
