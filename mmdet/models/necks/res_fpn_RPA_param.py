import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from collections import OrderedDict
from ..registry import NECKS
from ..utils import ConvModule, _GlobalConvModule, _BoundaryRefineModule


# def conv2d(filter_in, filter_out, kernel_size, stride=1):
#     pad = (kernel_size - 1) // 2 if kernel_size else 0
#     return nn.Sequential(OrderedDict([
#         ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
#         ("bn", nn.BatchNorm2d(filter_out)),
#         ("relu", nn.LeakyReLU(0.1)),
#     ]))

# def make_five_conv(filters_list, in_filters):
#     m = nn.Sequential(
#         conv2d(in_filters, filters_list[0], 1),
#         conv2d(filters_list[0], filters_list[1], 3),
#         conv2d(filters_list[1], filters_list[0], 1),
#         conv2d(filters_list[0], filters_list[1], 3),
#         conv2d(filters_list[1], filters_list[0], 1),
#     )
#     return m

@NECKS.register_module
class RRAFPNP(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 cls_num=16,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(RRAFPNP, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cls_num = cls_num
        self.num_ins = len(in_channels)   # 4
        self.num_outs = num_outs   # 5
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.kernel_size = 11
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        if end_level == -1:
            self.backbone_end_level = self.num_ins  # self.backbone_end_level = 4
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level  # 1
        self.end_level = end_level  # -1
        self.add_extra_convs = add_extra_convs  # True
        self.extra_convs_on_inputs = extra_convs_on_inputs  # True

        # self.make_five_conv_use = self.make_five_conv([self.out_channels, self.out_channels*2], self.out_channels*2)
        self.make_five_conv_use = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.gcn_convs = nn.ModuleList()
        self.br_convs = nn.ModuleList()
        self.l1_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            # RA________________________________________________________________________________________________________
            ra = self.make_five_conv(self.out_channels*2 + self.in_channels[i])
            # print("eerdfdfdfdsfdsfds", self.out_channels*2 + self.in_channels[i])
            # RA________________________________________________________________________________________________________
            l_conv = ConvModule(
                # self.cls_num,
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            l1_conv = ConvModule(
                in_channels[i],
                self.cls_num,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            gcn_conv = _GlobalConvModule(
                in_channels[i],
                self.cls_num,
                (self.kernel_size, self.kernel_size)
            )
            #####################################
            br_conv = _BoundaryRefineModule(
                self.cls_num
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.l1_convs.append(l1_conv)
            self.gcn_convs.append(gcn_conv)
            self.br_convs.append(br_conv)
            self.make_five_conv_use.append(ra)


        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def make_five_conv(self, in_filters):
        m = nn.Sequential(
            ConvModule(
                in_filters, 256, 1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                activation=self.activation,
                inplace=False),
            ConvModule(
                256, 64, 3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                activation=self.activation,
                inplace=False),
            ConvModule(
                64, 64, 1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                activation=self.activation,
                inplace=False),
            ConvModule(
                64, 64, 3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                activation=self.activation,
                inplace=False),
            ConvModule(
                64, 256, 1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                activation=self.activation,
                inplace=False))
        return m

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        l1_out_lays = [
            l1_conv(inputs[i + self.start_level]) for i, l1_conv in enumerate(self.l1_convs)
        ]
        # gcn___________________________________________________________________________________________________________
        gcn_out_lays = [
            gcn_conv(inputs[i + self.start_level]) for i, gcn_conv in enumerate(self.gcn_convs)
        ]
        # gcn___________________________________________________________________________________________________________

        # br____________________________________________________________________________________________________________
        br_out_lays = [
            br_conv(gcn_out_lays[i + self.start_level]) for i, br_conv in enumerate(self.br_convs)

        ]
        # br____________________________________________________________________________________________________________
        # +
        for i in range(len(br_out_lays)):
            br_out_lays[i] = l1_out_lays[i] + br_out_lays[i]
        # 1x1___________________________________________________________________________________________________________
        # print("-------------------------------------------------------------------------------------------")
        laterals = [
            # inputs[i + self.start_level]:backbone输出的fmp
            # lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)
            lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # 1x1___________________________________________________________________________________________________________
        # print("-------------------------------------------------------------------------------------------")
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            temp = F.interpolate(laterals[i], scale_factor=2, mode='nearest')
            # print(f"[INFO] laterals[{i}].shape:{laterals[i].shape}")
            # print(f"[INFO] temp.shape:{temp.shape}")
            laterals[i - 1] = torch.cat([laterals[i - 1], temp, inputs[i - 1]], axis=1)
            # print(f"[INFO] laterals[{i - 1}].shape:{laterals[i - 1].shape}")

            laterals[i - 1] = self.make_five_conv_use[i - 1](laterals[i - 1])
            # print(f"[INFO] laterals[{i - 1}].shape:{laterals[i - 1].shape}")
        # build outputs
        # part 1: from original levels
        # assert False
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
