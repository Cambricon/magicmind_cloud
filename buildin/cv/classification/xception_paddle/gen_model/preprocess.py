import numpy as np
import os
from PIL import Image
import logging
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from operators import DecodeImage, ResizeImage, CropImage, NormalizeImage

def imagenet_dataset(
    val_txt="/path/to/modelzoo/datasets/imageNet2012/labels.txt",
    image_file_path="/path/to/modelzoo/datasets/imageNet2012/",
    count=-1
):
    with open(val_txt, "r") as f:
        lines = f.readlines()
    logging.info("%d pictures will be read." % len(lines))
    current_count = 0
    for line in lines:
        image_name, label = line.split(" ")
        image_path = os.path.join(image_file_path, image_name)
        with open(image_path, 'rb') as f:
            img = f.read()
        yield img, label.strip()
        current_count += 1
        if current_count >= count and count != -1:
            break

def preprocess(input_image, dst_size=(299,299), transpose=True, normalize=True):
    resize_h, resize_w = (320, 320)
    crop_h, crop_w = dst_size
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    decodeimg = DecodeImage(to_rgb = True,to_np = True, channel_first = False)
    resizeimg = ResizeImage(size = None, resize_short = 320, interpolation=None, backend="cv2")
    cropimg =  CropImage(crop_h)
    normimg =  NormalizeImage(scale = 1/255, mean = mean, std =std, order = '')
    decodeout = decodeimg(input_image)
    resizeout = resizeimg(decodeout)
    cropout = cropimg(resizeout)
    if normalize:
        normout = normimg(cropout)
    else :
        normout = cropout

    input_numpy = normout
    if transpose:
        input_numpy = np.transpose(input_numpy, (2, 0, 1))
    return input_numpy

class Record:
    def __init__(self, filename):
        self.file = open(filename, "w+")

    def write(self, line, _print = False):
        self.file.write(line + "\n")
        if _print:
            print(line)

