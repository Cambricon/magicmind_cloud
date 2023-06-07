import numpy as np
import magicmind.python.runtime as mm
import cv2
from typing import List
import os
import glob

class CalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, max_samples: int, img_list):
        super().__init__()
        self.data_paths_ = glob.glob(img_list + "/images/*/*.jpg")
        # print("calibrate samples : ", len(self.data_paths_))
        self.shape_ = shape
        self.max_samples_ = min(max_samples, len(self.data_paths_))
        self.cur_sample_ = None
        self.cur_data_index_ = 0

    def get_shape(self):
        return self.shape_

    def get_data_type(self):
        return mm.DataType.FLOAT32

    def get_sample(self):
        return self.cur_sample_

    def letterbox(self, img):
        src_h, src_w = img.shape[0], img.shape[1]
        dst_h, dst_w = self.shape_.GetDimValue(2), self.shape_.GetDimValue(3)
        ratio = min(dst_h / src_h, dst_w / src_w)
        unpad_h, unpad_w = int(round(src_h * ratio)), int(round(src_w * ratio))
        img = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
        # padding
        pad_t = int(round((dst_h - unpad_h) / 2))
        pad_b = dst_h - unpad_h - pad_t
        pad_l = int(round((dst_w - unpad_w) / 2))
        pad_r = dst_w - unpad_w - pad_l
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        return img

    def preprocess_images(self, data_begin: int, data_end: int) -> np.ndarray:
        imgs = []
        for i in range(data_begin, data_end):
            img = cv2.imread(self.data_paths_[i])
            # resize as letterbox
            img = self.letterbox(img)
            img = np.float32(img)
            img -= (104, 117, 123)
            # HWC to CHW
            img = img.transpose(2, 0, 1)
            imgs.append(np.ascontiguousarray(img)[np.newaxis,:])
        # batch
        return np.ascontiguousarray(np.concatenate(tuple(imgs), axis=0).astype(dtype = np.float32))

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