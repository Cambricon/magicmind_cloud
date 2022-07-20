import numpy as np
from torchvision import transforms
import os
from PIL import Image
import logging

def imagenet_dataset(
    val_txt="/nfsdata/datasets/imageNet2012/labels.txt",
    image_file_path="/nfsdata/datasets/imageNet2012/",
    count=-1
):
    with open(val_txt, "r") as f:
        lines = f.readlines()
    logging.info("%d pictures will be read." % len(lines))
    current_count = 0
    for line in lines:
        image_name, label = line.split(" ")
        image_path = os.path.join(image_file_path, image_name)
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        yield img, label.strip()
        current_count += 1
        if current_count >= count and count != -1:
            break

def preprocess(input_image, transpose):
    resize_h, resize_w = (256, 256)
    crop_h, crop_w = (224, 224)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    preprocess = transforms.Compose(
        [
            transforms.Resize(resize_h),
            transforms.CenterCrop(crop_h),
            transforms.ToTensor(),
            normalize,
        ]
    )
    input_tensor = preprocess(input_image)
    input_numpy = input_tensor.numpy()
    if transpose:
        input_numpy = np.transpose(input_numpy, (1, 2, 0))
    return input_numpy

