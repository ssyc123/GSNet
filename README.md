
# Enhanced Neck Feature Representation for Object Detection in Aerial Images [arxiv](https://arxiv.org/abs/2204.02033)

Object detection in aerial images is a fundamental research topic in the geoscience and remote sensing domain. However, the advanced approaches on this topic mainly focus on designing the elaborate backbones or head networks but ignore neck networks. In this letter, we first underline the importance of the neck network in object detection from the perspective of information bottleneck. Then, to alleviate the information deficiency problem in the current approaches, we propose a global semantic
network (GSNet), which acts as a bridge from the backbone network to the head network in a bidirectional global pattern. Compared to the existing approaches, our model can capture the rich and enhanced image features with less computational costs. Besides, we further propose a feature fusion refinement module (FRM) for different levels of features, which are suffering from the problem of semantic gap in feature fusion. To demonstrate the effectiveness and efficiency of our approach, experiments are carried out on two challenging and representative aerial image datasets (i.e., DOTA and HRSC2016). Experimental results in terms of accuracy and complexity validate the superiority of our method. The code has been open-sourced at GSNet.

****

## Introduction
This codebase is created to build benchmarks for object detection in aerial images.
It is modified from [mmdetection](https://github.com/open-mmlab/mmdetection).
The master branch works with **PyTorch 1.1** or higher. If you would like to use PyTorch 0.4.1,
please checkout to the [pytorch-0.4.1](https://github.com/open-mmlab/mmdetection/tree/pytorch-0.4.1) branch.

## Results
Visualization results for oriented object detection on the test set of DOTA.
![Different class results](/show/fig4.png)

 Comparison to the baseline on DOTA for oriented object detection with ResNet-101. The figures with blue boxes are the results of the baseline and red boxes are the results of our proposed GSNet.
![Baseline and GSNet results](/show/fig3.png)

## Experiment

ImageNet Pretrained Model from Pytorch
- [ResNet50](https://drive.google.com/file/d/1mQ9S0FzFpPHnocktH0DGVysufGt4tH0M/view?usp=sharing)
- [ResNet101](https://drive.google.com/file/d/1qlVf58T0fY4dddKst5i7-CL3DXhBi3Mp/view?usp=sharing)

The effectiveness of our proposed methods with different backbone network on the test of DOTA.
|Backbone|Detector|+GS|Weight|mAP(%)|
|:---:|:---:|:---:|:---:|:---:|
|ResNet-101|Faster R-CNN||[download](https://github.com/ssyc123/GSNet/releases/download/v1.0/FastRCNN_DOTA_Baseline.pth)|73.09|
|ResNet-101|Faster R-CNN|+|[download](https://github.com/ssyc123/GSNet/releases/download/v1.0/FastRCNN_DOTA_GSNet.pth)|79.37|


GSNet Results in HRSC2016.
|Backbone|Detector|Weight|mAP(%)|
|:---:|:---:|:---:|:---:|
|ResNet-101|Faster R-CNN|[download](https://github.com/ssyc123/GSNet/releases/download/v1.0/FastRcnn_HRSC_GSNet.pth)|90.50|


## Installation

Please refer to [INSTALL.md](INSTALL.md) for installation.

    
## Get Started

Please see [GETTING_STARTED.md](GETTING_STARTED.md) for the basic usage of mmdetection.

## Contributing

We appreciate all contributions to improve benchmarks for object detection in aerial images. 


## Citing

If you use our work, please consider citing:

```

```

## Thanks to the Third Party Libs

[Pytorch](https://pytorch.org/)

[mmdetection](https://github.com/open-mmlab/mmdetection)

[AerialDetection](https://github.com/dingjiansw101/AerialDetection)
