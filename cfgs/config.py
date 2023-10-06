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

# @File    :   config.py
# @Time    :   2023/10/06 22:57:04
# @Author  :   Hanyu Feng
# @Version :   1.0
# @Contact :   feng.hanyu@wustl.edu
# @Description :

import os
import yaml
import torch
import argparse

class Dict2Class(object):
    def __init__(self, _obj) -> None:
        if _obj:
            for k, v in _obj.items():
                if type(v) == dict:
                    v = Dict2Class(v)
                # print("key:", k, ", value", v)
                self.__dict__[k] = v
                
def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="Mode for project train/eval")
    parser.add_argument("--config", type=str, default="./cfgs/default.yaml", help="Detailed configs")
    args = parser.parse_args()
    return args

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_config_copy_args(args):
    config_path = args.config
    assert True == os.path.exists(config_path), "Config file %s doesn't exist"%(config_path)
    
    cfg = yaml.load(open(config_path), Loader=yaml.FullLoader)
    cfg["Mode"] = args.mode
    cfg["Device"] = get_device()
    cfg = Dict2Class(cfg)
    print(f"Config version {cfg.Version}, device {cfg.Device}")
    return cfg
