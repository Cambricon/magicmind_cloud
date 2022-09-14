import numpy as np
import magicmind.python.runtime as mm
from magicmind.python.common.types import get_datatype_by_numpy
from typing import List

class CalibData(mm.CalibDataInterface):
    def __init__(self, data_list: List[np.ndarray]):
        super(CalibData, self).__init__()
        self._data_list = data_list
        self._cur_idx = 0
        status = self.reset()
        assert status.ok(), str(status)

    def next(self):
        if self._cur_idx >= len(self._data_list):
            return mm.Status(mm.Code.OUT_OF_RANGE, "No more data.")
        self._data = np.ascontiguousarray(self._data_list[self._cur_idx])
        self._data_shape = mm.Dims(self._data.shape)
        self._data_type = get_datatype_by_numpy(self._data.dtype)
        self._cur_idx += 1
        return mm.Status()

    def get_shape(self):
        return self._data_shape

    def get_data_type(self):
        return self._data_type

    def get_sample(self):
        return self._data

    def reset(self):
        self._cur_idx = 0
        return mm.Status()
