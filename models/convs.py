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

# @File    :   convs.py
# @Time    :   2023/04/01 15:36:23
# @Author  :   Hanyu Feng
# @Version :   1.0
# @Contact :   feng.hanyu@wustl.edu
# @Description :



import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.mode):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, use_padding=True):
        super(DepthwiseSeparableConv).__init__()
        padding_size = 0
        if use_padding:
            padding_size = (kernel_size + 1) // 2
        operations = [
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding_size, groups=in_channels),

            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1, stride=1)
        ]

        self.dwsconv = nn.Sequential(*operations)

    def forward(self, x):
        return self.dwsconv(x)

