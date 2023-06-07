import numpy as np
import magicmind.python.runtime as mm
import cv2
import os
import glob
import math
import torch
from PIL import Image
from magicmind.python.common.types import get_datatype_by_numpy
from typing import List

class CalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, scale: float, max_samples: int, img_dir: str):
        super().__init__()
        print(img_dir)
        assert os.path.isdir(img_dir)
        self.data_paths_ = glob.glob(img_dir + '/*.jpg')
        t = glob.glob(img_dir + '/*.JPEG')
        self.mean = [0, 0, 0]
        self.std = [1.0, 1.0, 1.0]
        self.scale_ = scale
        self.shape_ = shape
        self.data_paths_ += t
        self.max_samples_ = min(max_samples, len(self.data_paths_))
        self.cur_sample_ = None
        self.cur_data_index_ = 0

    def get_shape(self):
        return self.shape_
    
    def get_data_type(self):
        return mm.DataType.FLOAT32

    def get_sample(self):
        return self.cur_sample_
    
    def preprocess(self, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        pil_img = pil_img.resize((newW, newH), resample = Image.BICUBIC)
        img = np.asarray(pil_img)
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))#hwc->chw
        img = np.expand_dims(img, 0)
        img = img / 255
        return img
    
    def preprocess_images(self, data_begin: int, data_end: int) -> np.ndarray:
        imgs = []
        for i in range(data_begin, data_end):
            img = Image.open(self.data_paths_[i])
            img = self.preprocess(img, self.scale_) 
            imgs.append(img)
        return np.ascontiguousarray(np.concatenate(tuple(imgs),axis=0))

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
