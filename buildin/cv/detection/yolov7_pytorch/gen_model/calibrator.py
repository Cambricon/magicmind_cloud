import numpy as np
import magicmind.python.runtime as mm
import cv2
import os
import glob
import math
import logging
from PIL import Image

def coco_dataset(
    file_list_txt="coco_file_list_5000.txt",
    image_dir="../../../../datasets/coco/val2017",
    count=-1
):
    with open(file_list_txt, "r") as f:
        lines = f.readlines()
    logging.info("%d pictures will be read." % len(lines))
    current_count = 0
    for line in lines:
        image_name = line.strip()
        image_path = os.path.join(image_dir, image_name)
        img = cv2.imread(image_path)
        yield img, image_path
        current_count += 1
        if current_count >= count and count != -1:
            break

def letterbox(img, dst_shape):
    src_h, src_w = img.shape[0], img.shape[1]
    dst_h, dst_w = dst_shape
    ratio = min(dst_h / src_h, dst_w / src_w)
    unpad_h, unpad_w = int(math.floor(src_h * ratio)), int(math.floor(src_w * ratio))
    if ratio != 1:
        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (unpad_w, unpad_h), interp)
    # padding
    pad_t = int(math.floor((dst_h - unpad_h) / 2))
    pad_b = dst_h - unpad_h - pad_t
    pad_l = int(math.floor((dst_w - unpad_w) / 2))
    pad_r = dst_w - unpad_w - pad_l
    img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(114,114,114))
    return img, ratio

def preprocess_image(img, dst_shape) -> np.ndarray:
    # resize as letterbox
    img, ratio = letterbox(img, dst_shape)
    # BGR to RGB, HWC to CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    # normalize
    img = img.astype(dtype = np.float32) / 255.0
    return img

class CalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, max_samples: int, img_dir: str):
        super().__init__()
        # print(img_dir)
        assert os.path.isdir(img_dir)
        self.data_paths_ = glob.glob(img_dir + '/*.jpg')
        self.shape_ = shape
        self.max_samples_ = min(max_samples, len(self.data_paths_))
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        self.dst_shape_ = (self.shape_.GetDimValue(2), self.shape_.GetDimValue(3))

    def get_shape(self):
        return self.shape_

    def get_data_type(self):
        return mm.DataType.FLOAT32

    def get_sample(self):
        return self.cur_sample_
    
    def preprocess_images(self, data_begin: int, data_end: int) -> np.ndarray:
        imgs = []
        for i in range(data_begin, data_end):
            img = cv2.imread(self.data_paths_[i])
            img = preprocess_image(img, self.dst_shape_)
            imgs.append(img[np.newaxis,:])
        # batch and normalize
        return np.ascontiguousarray(np.concatenate(tuple(imgs), axis=0))

    def next(self):
        batch_size = self.shape_.GetDimValue(0)
        data_begin = self.cur_data_index_
        data_end = data_begin + batch_size
        if data_end > self.max_samples_:
            return mm.Status(mm.Code.OUT_OF_RANGE, "Data end reached")
        self.cur_sample_ = self.preprocess_images(data_begin, data_end)
        self.cur_data_index_ = data_end
        return mm.Status.OK()

    def reset(self):
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        return mm.Status.OK()
