import numpy as np
import cv2
import os

class Record:
    def __init__(self, filename):
        self.file = open(filename, "w")

    def write(self, line, _print = False):
        self.file.write(line + "\n")
        if _print:
            print(line)
            
def letterbox(img, dst_shape):
    src_h, src_w = img.shape[0], img.shape[1]
    dst_h, dst_w = dst_shape
    ratio = min(dst_h / src_h, dst_w / src_w)
    if ratio != 1:
        img = cv2.resize(img, (dst_w, dst_h))
    return img, ratio

def load_images(image_dir,bs):
    images = []
    for _file in os.listdir(image_dir):
        if ".jpg" or ".JPEG" in _file:
            images.append(_file)
    patch = bs - len(images) % bs
    for i in range(patch):
        images.append(images[i])
    return images

def preprocess_img(src_img,img_size):
    # convert to float32
    src_img = src_img.astype(dtype = np.float32)
    
    # resize
    img,ratio = letterbox(src_img,(img_size[1],img_size[2]))
    
    # bgr2rgb
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    # norm
    std = (58.395, 57.12, 57.375)
    mean = (123.675, 116.28, 103.53)
    img = (img-mean)/std
    
    # transpose
    return np.transpose(img,(2,0,1)),ratio