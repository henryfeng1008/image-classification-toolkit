# Copyright 2023 Hanyu Feng

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @File    :   FCOS_net.py
# @Time    :   2023/05/14 14:01:19
# @Author  :   Hanyu Feng
# @Version :   1.0
# @Contact :   feng.hanyu@wustl.edu
# @Description :

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import *
from fpn import *
from modules import *


__all__ = ["build_fcos"]


class FCOS_head(nn.Module):
    def __init__(self, feature_list, in_channels, out_channels) -> None:
        super().__init__()
        self.feature_list = feature_list
        self.out_channels = in_channels
        self.out_channels = out_channels
        self.share_heads_1 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels,
                       kernel_size=3),
            ConvBNReLU(in_channels=out_channels, out_channels=out_channels,
                       kernel_size=3),
            ConvBNReLU(in_channels=out_channels, out_channels=out_channels,
                       kernel_size=3),
            ConvBNReLU(in_channels=out_channels, out_channels=out_channels,
                       kernel_size=3),
        )
        self.share_heads_2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels,
                       kernel_size=3),
            ConvBNReLU(in_channels=out_channels, out_channels=out_channels,
                       kernel_size=3),
            ConvBNReLU(in_channels=out_channels, out_channels=out_channels,
                       kernel_size=3),
            ConvBNReLU(in_channels=out_channels, out_channels=out_channels,
                       kernel_size=3),
        )
        self.cls_head = nn.Conv2d(in_channels=out_channels,
                                    out_channels=2,
                                    kernel_size=3,
                                    padding=1)
        self.bbox_reg_head = nn.Conv2d(in_channels=out_channels,
                                    out_channels=4,
                                    kernel_size=3,
                                    padding=1)
        self.ctrness_head = nn.Conv2d(in_channels=out_channels,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1)

    def forward(self, x):
        cls_out, bbos_reg_out, ctrness_out = [], [], []
        for key in self.feature_list:
            feature = x[key]

            feature_1 = self.share_heads_1(feature)
            cls_out.append(self.cls_head(feature_1))
            ctrness_out.append(self.ctrness_head(feature_1))

            feature_2 = self.share_heads_2(feature)
            bbos_reg_out.append(self.bbox_reg_head(feature_2))

        return cls_out, bbos_reg_out, ctrness_out


class FCOS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = build_resnet18(pretrain=True)

        fpn_feature_list = {"C3": 128, "C4": 256, "C5":512}
        self.fpn = build_fpn(feature_list=fpn_feature_list,
                             out_channels=128)

        fcos_feature_list = ["C3"]
        self.fcos_head = FCOS_head(feature_list=fcos_feature_list,
                                   in_channels=128,
                                   out_channels=128).to("cuda")

    def forward(self, x):
        features = self.backbone(x)
        features.update(self.fpn(features))
        cls_out, bbox_reg_out, centerness_out = self.fcos_head(features)
        return cls_out, bbox_reg_out, centerness_out

def build_fcos():
    model = FCOS()
    return model

if __name__ == "__main__":
    myModel = FCOS()
    fake_input = torch.ones((4, 3, 224, 224))
    fake_input = fake_input.to("cuda")
    cls_out, bbox_reg_out, ctrness_out = myModel(fake_input)
    for cls_result in cls_out:
        print(f"cls_result:{cls_result.shape}")
    for bbox_reg_result in bbox_reg_out:
        print(f"bbox_reg_result:{bbox_reg_result.shape}")
    for ctrness_result in ctrness_out:
        print(f"ctrness_result:{ctrness_result.shape}")
