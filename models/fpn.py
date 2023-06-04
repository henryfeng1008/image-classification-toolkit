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

# @File    :   Resnet.py
# @Time    :   2023/05/14 21:52:19
# @Author  :   Hanyu Feng
# @Version :   1.0
# @Contact :   feng.hanyu@wustl.edu
# @Description :

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# __all__ = ["build_fpn"]


class FPN(nn.Module):
    def __init__(self, feature_list, out_channels, norm_layer=None) -> None:
        super().__init__()
        self.feature_list = feature_list

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._normal_layer = norm_layer

        in_conv = []
        out_conv = []
        for key in feature_list.keys():
            in_channels = feature_list[key]
            in_conv.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1))
            out_conv.append(nn.Conv2d(in_channels=out_channels,
                                      out_channels=out_channels,
                                      kernel_size=3, padding=1))
        self.in_conv = nn.Sequential(*in_conv)
        self.out_conv = nn.Sequential(*out_conv)

    def forward(self, x):
        output = OrderedDict()
        key_list = list(self.feature_list.keys())
        key = key_list[-1]

        out_key = "P" + key[1:]
        cur_output = self.in_conv[-1](x[key])
        prev_out = self.out_conv[-1](cur_output)
        output[out_key] = prev_out

        for key_idx in range(len(key_list) - 2, -1, -1):
            key = key_list[key_idx]
            out_key = "P" + key[1:]
            cur_output = self.in_conv[key_idx](x[key])
            cur_output += F.interpolate(prev_out,
                                       scale_factor=2)

            prev_out = self.out_conv[key_idx](cur_output)
            output[out_key] = prev_out
        return output

def build_fpn(feature_list, out_channels):
    model = FPN(feature_list, out_channels).to("cuda")
    return model

if __name__ == "__main__":
    c3 = torch.randn((4, 128, 28, 28))
    c4 = torch.randn((4, 256, 14, 14))
    c5 = torch.randn((4, 512, 7, 7))
    in_feature = {"C3": c3, "C4": c4, "C5":c5}
    feature_list = {"C3": 128,
                    "C4": 256,
                    "C5": 512}
    model = FPN(feature_list, 128)
    print(model)
    out = model(in_feature)
    for key in out.keys():
        print(key, out[key].shape)
