
# Learning to Reduce Information Bottleneck for Object Detection in Aerial Images [arxiv](https://arxiv.org/abs/2204.02033v2)

Object detection in aerial images is a critical and essential task in the fields of geoscience and remote sensing. Despite the popularity of computer vision methods in detecting objects in aerial images, these methods have been faced with significant limitations such as appearance occlusion and variable image sizes. In this letter, we explore the limitations of conventional neck networks in object detection by analyzing information bottlenecks. We propose an enhanced neck network to address the information deficiency issue in current neck networks. Our proposed neck network serves as a bridge between the backbone network and the head network. The enhanced neck network comprises a global semantic network (GSNet) and a feature fusion refinement module (FRM). The GSNet is designed to perceive contextual surroundings and propagate discriminative knowledge through a bidirectional global pattern. The FRM is developed to exploit different levels of features to capture comprehensive location information. We validate the efficacy and efficiency of our approach through experiments conducted on two challenging datasets, DOTA and HRSC2016. Our method outperforms existing approaches in terms of accuracy and complexity, demonstrating the superiority of our proposed method. The code has been open-sourced at GSNet.

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
@article{shen2022learning,
  title={Learning to reduce information bottleneck for object detection in aerial images},
  author={Shen, Yuchen and Song, Zhihao and Fu, Liyong and Jiang, Xuesong and Ye, Qiaolin},
  journal={arXiv preprint arXiv:2204.02033},
  year={2022}
}
```

## Thanks to the Third Party Libs

[Pytorch](https://pytorch.org/)

[mmdetection](https://github.com/open-mmlab/mmdetection)

[AerialDetection](https://github.com/dingjiansw101/AerialDetection)
