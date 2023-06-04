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

# @File    :   misc_utils.py
# @Time    :   2023/05/14 14:01:36
# @Author  :   Hanyu Feng
# @Version :   1.0
# @Contact :   feng.hanyu@wustl.edu
# @Description :


import os
import torch
import yaml


def load_config(config_path, check=False):
    f = open(config_path)
    config = yaml.load(f, Loader=yaml.FullLoader)
    if check:
        print(config)
    return config

def image_normalization(img):
    PIXEL_MEAN = [103.939, 116.779, 123.68]
    PIXEL_STD = [57.375, 57.12, 58.393]
    pixel_mean = torch.tensor(PIXEL_MEAN).view(-1, 1, 1).to(img.device)
    pixel_std = torch.tensor(PIXEL_STD).view(-1, 1, 1).to(img.device)
    img = (img - pixel_mean) / pixel_std
    return img