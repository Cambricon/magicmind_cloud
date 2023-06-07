import cv2
import numpy as np
import os
from PIL import Image
import logging
from torchvision import transforms

def imagenet_dataset(
    val_txt,
    image_file_path,
    count=-1
):
    with open(val_txt, "r") as f:
        lines = f.readlines()
    logging.info("%d pictures will be read." % len(lines))
    if len(lines) < count:
        print("infer pictures less than {}".format(count))
        return
    current_count = 0
    for line in lines:
        image_name, label = line.split(" ")
        image_path = os.path.join(image_file_path, image_name)
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        yield np.array(img), label.strip()
        current_count += 1
        if current_count >= count and count != -1:
            break

def preprocess(img, transpose, normalization):
    img = cv2.resize(img, (256, 256))
    h, w = img.shape[0], img.shape[1]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img = img[y0 : y0+224, x0 : x0+224, :]
    if normalization:
        img = img / 255.
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = img.astype(np.float32)
    if not transpose:
        img = np.transpose(img, axes=[2, 0, 1])
    return img
