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
        self.std = np.sqrt(vars_)

    def get_shape(self):
        return self.shape_

    def get_data_type(self):
        return mm.DataType.FLOAT32

    def get_sample(self):
        return self.cur_sample_

    def preprocess_images(self, data_begin: int, data_end: int) -> np.ndarray:
        imgs = []
        dst_h, dst_w = self.shape_.GetDimValue(2), self.shape_.GetDimValue(3)
        for i in range(data_begin, data_end):
            img = cv2.imread(self.data_paths_[i])
            # resize to NEW_SIZE
            img = cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
            # mean std
            img = img.astype(np.float32)
            MEAN = [0, 0, 0]
            STD = [1.0, 1.0, 1.0]
            img -= MEAN
            img /= STD
            # HWC to CHW
            img = img.transpose(2, 0, 1)
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
