import os
import cv2
import numpy as np


def get_data(anno_dir):
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
            item["file_name"] = file_name

            line_idx += 1
            face_num = int(lines[line_idx].strip())
            # print(face_num, type(face_num))
            if face_num > max_face_num:
                max_face_num = face_num
            item["face_num"] = face_num

            bbox = []
            for face_idx in range(face_num):
                line_idx += 1
                box = lines[line_idx].strip()
                box = box.split(" ")
                for item_idx in range(len(box)):
                    box[item_idx] = int(box[item_idx])
                bbox.append(box)
            item["bbox"] = bbox

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
    anno = get_data(r"../../data/Wider_face/wider_face_split/wider_face_split/wider_face_train_bbx_gt.txt")
    check = anno[0]
    print(f"sample anno: {check}")
    print(f"image num: {len(anno)}")

    data_root = r"../../data/Wider_face/WIDER_train/WIDER_train/images"
    for idx in range(10):
        check = anno[idx]
        file_name = os.path.join(data_root, check['file_name'])
        bbox = check['bbox']
        plot_image(file_name, bbox)


if __name__ == "__main__":
    main()
