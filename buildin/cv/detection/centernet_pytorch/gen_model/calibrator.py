import numpy as np
import cv2
from typing import List
import os
import glob
import math
import magicmind.python.runtime as mm
def preprocess_image(img, dst_shape) -> np.ndarray:
    # resize
    img = cv2.resize(img, (dst_shape[1], dst_shape[0]), cv2.INTER_LINEAR)
    # BGR to RGB
    img = img[:, :, ::-1]
    # normalize
    MEAN = [0.40789654 * 255, 0.44719302 * 255, 0.47026115 * 255]
    STD = [0.28863828 * 255, 0.27408164 * 255, 0.27809835 * 255]
    img = (img - MEAN) / STD
    # HWC to CHW
    img = img.astype(dtype = np.float32).transpose(2, 0, 1)
    return img

class CalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, max_samples: int, img_dir: str):
        super().__init__()
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
