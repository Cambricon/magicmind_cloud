import numpy as np
import magicmind.python.runtime as mm
from magicmind.python.common.types import get_datatype_by_numpy
from typing import List

class CalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, max_samples: int, sample_data: List):
        super().__init__()
        self.sample_data = sample_data
        self.shape_ = shape
        self.max_samples_ = min(max_samples, len(self.sample_data))
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        self.dst_shape_ = (self.shape_.GetDimValue(2), self.shape_.GetDimValue(3))
        
    def get_shape(self):
        return mm.Dims(self.cur_sample_.shape)

    def get_data_type(self):
        return mm.DataType.FLOAT32
    
    def get_sample(self):
        return self.cur_sample_
    
    def preprocess_images(self, data_begin: int, data_end: int) -> np.ndarray:
        imgs = []
        for i in range(data_begin, data_end):
            img = self.sample_data[i]
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
    
    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files