from .resnet import ResNet, make_res_layer
# from .resnet1 import ResNet, make_res_layer
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .hrnet import HRNet
# from .aff_resnet import ResNet

__all__ = ['ResNet', 'make_res_layer', 'ResNeXt', 'SSDVGG', 'HRNet']
