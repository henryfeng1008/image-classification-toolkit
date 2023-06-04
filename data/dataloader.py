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


from typing import Any, Iterator
import torch
import os
import cv2
import numpy as np
import json
import copy
import math
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class myDataSet(Dataset):
    def __init__(self, anno_file) -> None:
        super().__init__()
        with open(anno_file, 'r') as f:
            data_list = json.load(f)
        self.data_list = data_list

    def __getitem__(self, index):
        data_item = copy.deepcopy(self.data_list[index])
        return data_item

    def __len__(self):
        return len(self.data_list)

class MyDataMapper():
    def __init__(self):
        pass

    def __call__(self, data_item) -> Any:
        for item in data_item:
            src_image = cv2.imread(item['file_name'])
            ds_image = cv2.resize(src_image, (224, 224), cv2.INTER_AREA)
            ds_image = torch.as_tensor(np.ascontiguousarray(ds_image.transpose(2, 0, 1)))
            item["input_image"] = ds_image
        return data_item


def build_train_loader(anno_file, batch_size=16):
    det_dataset = myDataSet(anno_file=anno_file)
    mapper = MyDataMapper()
    dataloader = DataLoader(dataset=det_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0,
                            collate_fn=mapper,
                            drop_last=True)
    return dataloader


if __name__ == "__main__":
    anno_file = r'./data/anno/train_det_face_anno.json'
    my_loader = build_train_loader(anno_file)
    print(len(my_loader))

    for i, value in enumerate(my_loader):
        print(value)
        break
    # for idx in range(len(my_loader)):
    #     item = next(my_loader)
    #     print(item.keys())
    #     break
