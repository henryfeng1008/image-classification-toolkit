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

# @File    :   datasets.py
# @Time    :   2023/10/06 22:18:37
# @Author  :   Hanyu Feng
# @Version :   1.0
# @Contact :   feng.hanyu@wustl.edu
# @Description :

from typing import Any
import torch
from torch.utils.data import Dataset


class DefaultDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)
    
    def __len__(self):
        return 0
    
def build_dataset(data_list):
    print(__name__, data_list)
    return None
    pass