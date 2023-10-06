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

# @File    :   main.py
# @Time    :   2023/04/01 15:51:12
# @Author  :   Hanyu Feng
# @Version :   1.0
# @Contact :   feng.hanyu@wustl.edu
# @Description :


import os
import yaml
import torch
import argparse
import numpy as np
from data import build_dataloader
from cfgs import Dict2Class, build_args, get_device, load_config_copy_args
            

def main():
    args = build_args()
    print("[*] ", args)
    cfg = load_config_copy_args(args)

    if cfg.Mode == "train":
        print(f"cfg.Data.Version {cfg.Data.Version}")
        train_loader = build_dataloader(cfg.Data.TrainData, cfg.Solver.BatchSize)
        test_loader = build_dataloader(cfg.Data.ValidData, 1)
        
        model = build_model().to(cfg.Device)
        
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        # if cfg["Solver"]["Optimizer"] == "AdamW":
        #     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # for ep in range(cfg["Solver"])
        
        pass
    else:
        pass
    
    pass


if __name__ == "__main__":
    main()