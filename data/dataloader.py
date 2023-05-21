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

# @File    :   dataloader.py
# @Time    :   2023/05/21 21:35:58
# @Author  :   Hanyu Feng
# @Version :   1.0
# @Contact :   feng.hanyu@wustl.edu
# @Description :


import torch
import os
import cv2
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader


class myDataSet(Dataset):
    def __init__(self, anno_file) -> None:
        super().__init__()
        with open(anno_file, 'r') as f:
            data_dict = json.load(f)
        self.data_dict = data_dict

    def __getitem__(self, index):
        data_item = self.data_dict[index]

        src_image = cv2.imread(data_item['file_name'])
        data_item["input_data"] = src_image
        return data_item

    def __len__(self):
        return len(self.data_dict)



def build_train_loader(anno_file):
    det_dataset = myDataSet(anno_file=anno_file)
    dataloader = DataLoader(dataset=det_dataset,
                            batch_size=16,
                            shuffle=True,
                            num_workers=10,
                            drop_last=True)
    return dataloader


if __name__ == "__main__":
    anno_file = r'./data/anno/train_det_face_anno.json'
    my_loader = build_train_loader(anno_file)
    print(len(my_loader))
    for item in my_loader:
        print(item)
        break
    # for idx in range(len(my_loader)):
    #     item = next(my_loader)
    #     print(item.keys())
    #     break
