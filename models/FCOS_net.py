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
from modules import *


class FCOS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = build_resnet18(pretrain=True, verify=True)
        # self.FPN = FPN()

    def forward(self, x):
        # features = self.backbone(x)
        # detect_result = self.
        return self.backbone(x)

if __name__ == "__main__":
    myModel = FCOS()
    fake_input = torch.ones((4, 3, 224, 224))
    fake_input = fake_input.to("cuda")
    output = myModel(fake_input)