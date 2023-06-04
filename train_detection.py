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


import os
import torch
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# import sys
# sys.path.append("")
# import models.base_ops
from models.fcos import build_fcos
from data.dataloader import build_train_loader
from utils.misc_utils import image_normalization

if __name__ == "__main__":
    epoch = 10000
    anno_file = r'./data/anno/train_det_face_anno.json'
    train_data_loader = build_train_loader(anno_file, batch_size=32)
    detector = build_fcos()
    detector = detector.to("cuda")

    optimizer = optim.SGD(detector.parameters(), lr=1e-3)
    scheduler = CosineAnnealingWarmRestarts(
                        optimizer=optimizer,
                        T_0=epoch*len(train_data_loader) // 2,
                        eta_min=1e-6)

    print("Start training")
    print(f"{len(train_data_loader)} per epoch")
    for ep in range(epoch):
        for idx, data in enumerate(train_data_loader):
            input_image = [x["input_image"].to("cuda") for x in data]
            input_image = [image_normalization(x) for x in input_image]
            input_image = torch.stack(input_image)

            # gt_instance = data["gt_instance"]
            # print(gt_instance)
            optimizer.zero_grad()
            cls_result, bbox_reg_result, ctrness_result = detector(input_image)
            for cls_res in cls_result:
                print(f"cls_result:{cls_res.shape}")
            for bbox_reg_res in bbox_reg_result:
                print(f"bbox_reg_result:{bbox_reg_res.shape}")
            for ctrness_res in ctrness_result:
                print(f"ctrness_result:{ctrness_res.shape}")
            print(idx)
        print(ep)
            # loss = process_det_loss(cls_result,
            #                         bbox_reg_result,
            #                         ctrness_result,
            #                         gt_instance)
            # loss.backward()
            # optimizer.step()
            # scheduler.step(ep * len(train_data_loader) + idx)

