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

# @File    :   wider_face_data.py
# @Time    :   2023/05/14 14:01:29
# @Author  :   Hanyu Feng
# @Version :   1.0
# @Contact :   feng.hanyu@wustl.edu
# @Description :


import os
import cv2
import numpy as np
import json
import imagesize

def get_data(anno_dir, data_root):
    print(anno_dir)
    if os.path.exists(anno_dir):
        print("Path valid")
    else:
        print("Path invalid, please check again")
        return None

    f = open(anno_dir, "r")
    lines = f.readlines()
    anno = []
    line_idx = 0
    max_face_num = 0
    while line_idx < len(lines):
        if '.jpg'in lines[line_idx]:
            item = dict()
            file_name = lines[line_idx].strip()
            # print(file_name, type(file_name))
            file_name = os.path.join(data_root, file_name)
            item["file_name"] = file_name
            # print(file_name)
            width, height = imagesize.get(file_name)
            item["img_width"] = width
            item["img_height"] = height

            line_idx += 1
            face_num = int(lines[line_idx].strip())
            # print(face_num, type(face_num))
            if face_num > max_face_num:
                max_face_num = face_num
            item["face_num"] = face_num

            gt_instance = []
            for face_idx in range(face_num):
                instance = dict()
                line_idx += 1
                box = lines[line_idx].strip()
                box = box.split(" ")[:4]
                # for item_idx in range(len(box)):
                for item_idx in range(4):
                    box[item_idx] = int(box[item_idx])
                    if item_idx == 0 or item_idx == 2:
                        box[item_idx] = box[item_idx] / item["img_width"]
                    else:
                        box[item_idx] = box[item_idx] / item["img_height"]
                instance['bbox'] = box
                instance['class'] = 1
                gt_instance.append(instance)
            item["gt_instance"] = gt_instance

            anno.append(item)
            line_idx += 1
            # break
        else:
            line_idx += 1
    return anno


def plot_image(img_path, bbox):
    print(img_path)
    if os.path.exists(img_path):
        print("Path valid")
    else:
        print("Path invalid, please check again")
        return None
    img = cv2.imread(img_path)
    if len(bbox) != 0:
        for box in bbox:
            x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose = box
            if blur > 2 or occlusion > 10 or invalid != 0:
                continue
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("Sample", img)
    cv2.waitKey(0)


def main():
    wider_face_anno_dir = r"../../data/Wider_face/wider_face_split/wider_face_split/wider_face_train_bbx_gt.txt"
    data_root = r"../../data/Wider_face/WIDER_train/WIDER_train/images"

    my_anno_file = './data/anno/train_det_face_anno.json'
    anno = get_data(wider_face_anno_dir, data_root)
    check = anno[0]
    print(f"sample anno: {check}")
    print(f"image num: {len(anno)}")

    with open(my_anno_file, 'w') as f:
        json.dump(anno, f)
        f.close()

    # for idx in range(10):
    #     check = anno[idx]
    #     file_name = os.path.join(data_root, check['file_name'])
    #     bbox = check['bbox']
    #     plot_image(file_name, bbox)


if __name__ == "__main__":
    main()
