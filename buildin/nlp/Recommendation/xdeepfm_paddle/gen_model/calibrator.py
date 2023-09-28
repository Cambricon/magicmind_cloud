import magicmind.python.runtime as mm
import numpy as np

class CalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, index: int, max_samples: int, input_data: list):
        super().__init__()
        self.shape_ = shape
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        self.max_samples_ = max_samples
        self.index = index
        self.input_data = input_data

    def get_shape(self):
        return self.shape_

    def get_data_type(self):
        if self.index < 26:
            return mm.DataType.INT32
        else:
            return mm.DataType.FLOAT32

    def get_sample(self):
        return self.cur_sample_
    
    def preprocess(self, input_data: list) -> np.ndarray:
    
        if self.index < 26:
            input_data[self.index].astype(np.int32)
        return input_data[self.index]

    def next(self):
        batch_size = self.shape_.GetDimValue(0)
        data_begin = self.cur_data_index_
        data_end = data_begin + batch_size
        if data_end > self.max_samples_:
            return mm.Status(mm.Code.OUT_OF_RANGE, "Data end reached")
        self.cur_sample_ = self.preprocess(self.input_data)
        self.cur_data_index_ = data_end
        return mm.Status.OK()

    def reset(self):
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        return mm.Status.OK()