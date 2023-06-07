import numpy as np
import cv2
import magicmind.python.runtime as mm
from typing import List
import os
import glob


class CalibData(mm.CalibDataInterface):
    def __init__(
        self,
        shape: mm.Dims,
        max_samples: int,
        img_dir: str,
        means_: List[float],
        vars_: List[float],
    ):
        super().__init__()
        assert os.path.isdir(img_dir)
        self.data_paths_ = glob.glob(img_dir + "/*.jpg")
        t = glob.glob(img_dir + "/*.JPEG")
        self.data_paths_ += t
        self.shape_ = shape
        self.max_samples_ = min(max_samples, len(self.data_paths_))
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        self.means_ = means_
        self.std = np.sqrt(vars_)*1.0

    def get_shape(self):
        return self.shape_

    def get_data_type(self):
        return mm.DataType.FLOAT32

    def get_sample(self):
        return self.cur_sample_

    def preprocess_images(self, data_begin: int, data_end: int) -> np.ndarray:
        imgs = []
        img_h, img_w = self.shape_.GetDimValue(2), self.shape_.GetDimValue(3)
        for i in range(data_begin, data_end):
            img = cv2.imread(self.data_paths_[i])

            scaling_factor = min((img_h - 1) / (img.shape[0] - 1), (img_w - 1) / (img.shape[1] - 1))
            m = np.zeros((2, 3))
            m.astype('float64')
            m[0, 0] = scaling_factor
            m[1, 1] = scaling_factor
            img = cv2.warpAffine(img, m, (img_w, img_h),
                    flags = cv2.INTER_CUBIC if scaling_factor > 1.0 else cv2.INTER_AREA,
                    borderMode = cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            img = img.astype('float32')
            img -= self.means_
            img /= self.std 
            img = np.transpose(img, (2, 0, 1)) # HWC >>> CHW
            # ori calibrator will return image
            #return img 

            # cambricon-note: always use insert_bn, therefore we need to minus means and div std
            imgs.append(np.ascontiguousarray(img)[np.newaxis, :])
        # batch
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
