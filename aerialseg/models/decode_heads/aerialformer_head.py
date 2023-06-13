import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.norm import build_norm_layer
from mmcv.cnn.bricks.activation import build_activation_layer

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class MDCDecoder(BaseDecodeHead):
    """
    Multi-scale decoder used in AerialFormer.
    """

    def __init__(self, interpolate_mode="bilinear", **kwargs):
        super().__init__(input_transform="multiple_select", **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        self.in_channels = list(reversed(self.in_channels))
        assert num_inputs == len(self.in_index)

        self.up_convs = nn.ModuleList()
        self.dilated_convs = nn.ModuleList()

        custom_params_list = [
            {
                # Deepest Layer
                "kernel": (3, 3, 3),
                "padding": (1, 2, 3),
                "dilation": (1, 2, 3),
            },
            {
                "kernel": (3, 3, 3),
                "padding": (1, 2, 3),
                "dilation": (1, 2, 3),
            },
            {
                "kernel": (3, 3, 3),
                "padding": (1, 2, 3),
                "dilation": (1, 2, 3),
            },
            {
                "kernel": (3, 3, 3),
                "padding": (1, 1, 1),
                "dilation": (1, 1, 1),
            },
            {
                "kernel": (1, 3, 3),
                "padding": (0, 1, 1),
                "dilation": (1, 1, 1),
            },
        ]

        for idx in range(len(self.in_channels)):
            if idx != 0:
                self.up_convs.append(
                    self.up_pooling(self.in_channels[idx - 1], self.in_channels[idx])
                )
            else:
                self.up_convs.append(nn.Identity())

            self.dilated_convs.append(
                nn.Sequential(
                    MDCBlock(
                        in_channels=self.in_channels[idx] * 2 ** (idx != 0),
                        out_channels=self.in_channels[idx],
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        custom_params=custom_params_list[idx],
                    ),
                    ConvModule(
                        in_channels=self.in_channels[idx],
                        out_channels=self.in_channels[idx],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                )
            )

        self.conv_seg = nn.Conv2d(
            self.in_channels[-1], self.out_channels, kernel_size=1
        )

    def forward(self, inputs):
        inputs = list(reversed(self._transform_inputs(inputs)))
        assert len(inputs) == len(
            self.in_index
        ), f"The length of inputs must be {len(self.in_index)}. This is for {len(self.in_index)} stages of backbone. But got {len(inputs)} stages"
        # self.show_feature_map(inputs)
        x = inputs[0]
        x = self.dilated_convs[0](x)

        for idx in range(1, len(inputs)):
            x = self.up_convs[idx](x)
            x = torch.cat([x, inputs[idx]], dim=1)
            x = self.dilated_convs[idx](x)

        out = self.cls_seg(x)
        return out

    def up_pooling(self, in_channels, out_channels, kernel_size=2, stride=2):
        # Func for up sample
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
            ),
            build_norm_layer(self.norm_cfg, out_channels)[1],  # SyncBN
            build_activation_layer(self.act_cfg),
        )


class MDCBlock(nn.Module):
    """
    This is a module for multi-dilated convolution layers
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        norm_cfg,
        act_cfg,
        custom_params={
            "kernel": (3, 3, 3),
            "padding": (3, 5, 7),
            "dilation": (3, 5, 7),
        },
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.custom_params = custom_params
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.kernel = self.custom_params["kernel"]
        self.paddings = self.custom_params["padding"]
        self.dilations = self.custom_params["dilation"]
        SPLIT_NUM = 3

        self.layers = nn.ModuleList()

        self.pre_conv_layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            bias=False,
        )
        quotient = self.in_channels // SPLIT_NUM
        reminder = self.in_channels % SPLIT_NUM
        sprit_channels = [quotient] * SPLIT_NUM
        if reminder == 1:
            sprit_channels[0] += 1
            sprit_channels[1] += 1
            sprit_channels[2] -= 1
        elif reminder == 2:
            sprit_channels[0] += 1
            sprit_channels[1] += 1
        for kernel, padding, dilation, channels in zip(
            *custom_params.values(), sprit_channels
        ):
            self.layers.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                )
            )

        self.fusion_layer = nn.Conv2d(
            in_channels=self.in_channels,  # equals to out_channels*2
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False,
        )
        self.norm = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.act = build_activation_layer(self.act_cfg)

    def forward(self, x):
        x_shape = x.shape
        x = self.pre_conv_layer(x)
        x1, x2, x3 = torch.chunk(x, 3, dim=1)

        assert (
            x1.shape[1] + x2.shape[1] + x3.shape[1] == x_shape[1]
        ), f"{x1.shape[1]} + {x2.shape[1]} + {x3.shape[1]} != {x_shape[1]}"

        x1 = self.layers[0](x1)
        x2 = self.layers[1](x2)
        x3 = self.layers[2](x3)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fusion_layer(x)

        return self.act(self.norm(x))
