import numpy as np
from torchvision import transforms
import os
from PIL import Image
import cv2
import logging

def imagenet_dataset(
    val_txt,
    image_file_path,
    count=-1
):
    with open(val_txt, "r") as f:
        lines = f.readlines()
    logging.info("%d pictures will be read." % len(lines))
    #print("imagenet_dataset",val_txt,lines)
    current_count = 1
    for line in lines:
        #print("imagenet_dataset1",line)
        image_name, label, enter = line.split(" ")
        image_path = os.path.join(image_file_path, image_name)
        img = cv2.imread(image_path)
        yield img, label.strip()
        current_count += 1
        if current_count > count and count != -1:
            break

def preprocess(input_image, transpose):
    resize_h, resize_w = (256, 256)
    crop_h, crop_w = (224, 224)
    # resize
    resized = cv2.resize(input_image, (resize_w, resize_h))
    # # center crop
    x = resized.shape[1] / 2 - crop_w / 2
    y = resized.shape[0] / 2 - crop_h / 2
    crop_img = resized[int(y):int(y+crop_h), int(x):int(x+crop_w)]
    if transpose:
        input_numpy = np.transpose(crop_img, (2, 0, 1))
    else: 
        input_numpy = crop_img
    return input_numpy

def load_name(imagenet_label_path):
    name_map = {}
    with open(imagenet_label_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        idx = line.split(" ")[0]
        name = " ".join(line.split(" ")[1:])
        name_map[int(idx)] = name.strip()
    return name_map