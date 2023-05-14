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
from modules import *
from collections import OrderedDict


class Resnet_18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage_num = 4
        self.repeat_num_per_stage = [2, 2, 2, 2]
        self.out_channels = [64, 128, 256, 512]
        self.stride_per_stage = [1, 2, 2, 2]

        assert self.stage_num == len(self.out_channels)
        self.main_branch = []
        self.main_branch_key = []

        stage_in_channel = self.out_channels[0]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=stage_in_channel,
                               kernel_size=7, stride=2, padding=(3, 3))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1))

        for stage_idx in range(self.stage_num):
            stage_out_channel = self.out_channels[stage_idx]
            cur_stage_repeat_num = self.repeat_num_per_stage[stage_idx]
            for repeat_num in range(cur_stage_repeat_num):
                if repeat_num != cur_stage_repeat_num - 1:
                    key_name = \
                    "stage_" + str(stage_idx + 1) + "_" + str(repeat_num + 1)
                else:
                    key_name = \
                    "C" + str(stage_idx + 1)
                if repeat_num == 0:
                    conv_ops = ResBlock_18(in_channels=stage_in_channel,
                                           out_channels=stage_out_channel,
                                           kernel_size=3,
                                           stride=self.stride_per_stage[stage_idx])
                else:
                    conv_ops = ResBlock_18(in_channels=stage_out_channel,
                                           out_channels=stage_out_channel,
                                           kernel_size=3, stride=1)

                self.main_branch.append(conv_ops)
                self.main_branch_key.append(key_name)

            stage_in_channel = stage_out_channel
        self.main_branch_ops = zip(self.main_branch_key, self.main_branch)
        # for ops in self.main_branch:
        #     print(ops)

    def forward(self, x):
        output = OrderedDict()
        x = self.conv1(x)
        x = self.max_pool_1(x)
        output['C0'] = x
        for (key, ops) in self.main_branch_ops:
            x = ops(x)
            output[key] = x
        for key in output.keys():
            print(output[key].shape, key)
        return output