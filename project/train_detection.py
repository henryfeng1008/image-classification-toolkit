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

# @File    :   train_detection.py
# @Time    :   2023/05/21 21:22:31
# @Author  :   Hanyu Feng
# @Version :   1.0
# @Contact :   feng.hanyu@wustl.edu
# @Description :

import torch
import torch.optim as optim
import os
import numpy as np
from models.fcos import build_fcos
from data.dataloader import build_train_loader



if __name__ == "__main__":
    anno_file = r'./data/anno/train_det_face_anno.json'
    train_data_loader = build_train_loader(anno_file)
    detector = build_fcos()

    optimizer = optim.SGD(detector.parameters(), lr=1e-3)

    epoch = 10000
    for ep in range(epoch):
        for idx, data in enumerate(train_data_loader):
            input_image = data["input_image"]
            gt_instance = data["gt_instance"]
            optimizer.zero_grad()
            cls_result, bbox_reg_result, ctrness_result = detector(data)

            loss = process_det_loss(cls_result,
                                    bbox_reg_result,
                                    ctrness_result,
                                    gt_instance)
            loss.backward()
            optimizer.step()

