import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..registry import NECKS
from ..utils import ConvModule, _GlobalConvModule, _BoundaryRefineModule, iAFF, DAF, AFF, MS_CAM

@NECKS.register_module
class RGBFPN_IAFF(nn.Module):

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
        super(RGBFPN_IAFF, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cls_num = cls_num
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.kernel_size = 11
        self.iaff = iAFF(channels=self.out_channels)

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs




        self.lateral_convs = nn.ModuleList()   # define 1x1 conv list(channel:16 -> 256)
        self.fpn_convs = nn.ModuleList()       # define fpn 3x3 conv list(channel:256 -> 256)
        self.gcn_convs = nn.ModuleList()       # define gcn(11x11) conv list[0:256 -> 16, 1:512 -> 16, 2:1024 -> 16, 3:2048 -> 16,]
        self.br_convs = nn.ModuleList()        # define br(channel:16 -> 16) conv list
        self.l1_convs = nn.ModuleList()        # define res1x1 conv list[0:256 -> 16, 1:512 -> 16, 2:1024 -> 16, 3:2048 -> 16,]


        for i in range(self.start_level, self.backbone_end_level):  # range(0, 4): 0 ,1 ,2 ,3
            l_conv = ConvModule(
                self.cls_num,   # 16
                out_channels,   # 256
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

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        l1_out_lays = [
            l1_conv(inputs[i + self.start_level]) for i, l1_conv in enumerate(self.l1_convs)
        ]
        # l1_out_lays = []
        # for i, l1_conv in enumerate(self.l1_convs):
        #     l1_out_lays.append(l1_conv(inputs[i + self.start_level]))



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
        laterals = [
            # inputs[i + self.start_level]:backbone输出的fmp
            # lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)
            lateral_conv(br_out_lays[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # 1x1___________________________________________________________________________________________________________

        # build top-down path
        # used_backbone_levels = len(laterals)
        # for i in range(used_backbone_levels - 1, 0, -1):
        #     laterals[i - 1] += F.interpolate(
        #         laterals[i], scale_factor=2, mode='nearest')


        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):  # (3 ,2, 1)
            temp = F.interpolate(laterals[i], scale_factor=2, mode='nearest')
            laterals[i - 1] = self.iaff(laterals[i - 1], temp)

        # build outputs
        # part 1: from original levels
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
