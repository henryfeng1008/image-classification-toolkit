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
from torchvision import models

# __all__ = ["build_resnet18"]

class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 norm_layer=None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3,stride=1,
                               padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Resnet18(nn.Module):
    def __init__(self, norm_layer=None) -> None:
        super().__init__()
        self.layers = [2, 2, 2, 2]
        self.in_channels = 64
        self.num_class = 1000

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._normal_layer = norm_layer

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=self.in_channels,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, 64, self.layers[0])
        self.layer2 = self._make_layer(ResBlock, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(ResBlock, 256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(ResBlock, 512, self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, repeat_num, stride=1):
        norm_layer = self._normal_layer
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                          kernel_size=1, stride=stride,
                          bias=False),
                norm_layer(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels,
                           stride, downsample, norm_layer))
        self.in_channels = out_channels
        for _ in range(1, repeat_num):
            layers.append(block(self.in_channels, out_channels,
                               norm_layer=norm_layer))
        return nn.Sequential(*layers)


    def forward(self, x):
        output = OrderedDict()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        output["C1"] = x
        x = self.layer1(x)
        output["C2"] = x
        x = self.layer2(x)
        output["C3"] = x
        x = self.layer3(x)
        output["C4"] = x
        x = self.layer4(x)
        output["C5"] = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return output

def build_resnet18(pretrain=False, verify=False):
    pretrain_weight = r'../../Pretrain_model/resnet18-f37072fd.pth'
    model = Resnet18()
    if pretrain:
        state_dict = torch.load(pretrain_weight)
        model.load_state_dict(state_dict, strict=True)
    model.to("cuda")

    if pretrain and verify:
        ref_resnet_18 = models.resnet18(pretrained=True)
        ref_resnet_18.to("cuda")
        input_tensor = torch.ones((1, 3, 224, 224), requires_grad=False)
        input_tensor = input_tensor.to("cuda")

        my_output, _ = model(input_tensor)
        ref_output = ref_resnet_18(input_tensor)
        if not my_output.equal(ref_output):
            raise ValueError("resnet 18 not parity with public version !!!")
        else:
            print("resnet 18 parity")
    return model



if __name__ == "__main__":
    pretrain_weight = r'../../Pretrain_model/resnet18-f37072fd.pth'
    device = torch.device("cuda")
    # model = torch.load(pretrain_weight)
    # print(model.keys())
    my_resnet_18 = Resnet18()
    my_resnet_18.load_state_dict(torch.load(pretrain_weight), strict=True)
    my_resnet_18.to(device)
    my_state_dict = my_resnet_18.state_dict()

    ref_resnet_18 = models.resnet18()
    ref_resnet_18.load_state_dict(torch.load(pretrain_weight), strict=True)
    ref_resnet_18.to(device)
    ref_state_dict = ref_resnet_18.state_dict()

    # print(f"my_state_dict {my_state_dict.keys()}")
    # print(f"ref_state_dict {ref_state_dict.keys()}")

    # for key in ref_state_dict.keys():
    #     ref_param = ref_state_dict[key]
    #     my_param = my_state_dict[key]
    #     print(key,"\t", my_param.equal(ref_param))

    input_tensor = torch.ones((1, 3, 224, 224), requires_grad=False)
    input_tensor = input_tensor.to(device)

    my_output = my_resnet_18(input_tensor)
    ref_output = ref_resnet_18(input_tensor)


    print(my_output)
    print(ref_output)
    # print(my_output.equal(ref_output))

    # print(my_resnet_18)
    # print(ref_resnet_18)

    # print(my_output)
    # print(ref_output)