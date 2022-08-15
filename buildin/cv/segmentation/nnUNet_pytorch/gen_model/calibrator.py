import magicmind.python.runtime as mm
from magicmind.python.common.types import get_datatype_by_numpy

class CalibData(mm.CalibDataInterface):
        def __init__(self, args):
            super().__init__()
            self.count = 1
            self.pth_path = args.calib_data_path

        def get_shape(self):
            return mm.Dims(self.data[0].shape)
        def get_data_type(self):
            return get_datatype_by_numpy(self.data[0].dtype)

        def get_sample(self):
            return self.data[0]

        def next(self):
            import torch
            if self.count:
                self.data = [torch.load(self.pth_path).numpy() ]
                self.count = self.count - 1
            else:
                return mm.Status(mm.Code.OUT_OF_RANGE, "No more data.")
            return mm.Status.OK()

        def reset(self):
            self.count = 1
            return mm.Status.OK()