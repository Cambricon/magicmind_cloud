import numpy as np
import time
import magicmind.python.runtime as mm

class MeasureTime():
    def __init__(self, measurements, key):
        self.measurements = measurements
        self.key = key

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.measurements[self.key] = time.perf_counter() - self.t0

def _check_status(status : mm.Status):
  assert status.ok(), str(status)

class NNModel:
    '''
    MagicMind Model wrapper, used to initialize runtime resources:
    create engine
    create context
    create input tensors
    create task queue

    Fill in the data at NNModel.input_tensors before calling NNModel.execute
    '''
    def __init__(self, device : mm.Device, mm_model : mm.Model):
        assert device is not None, 'device can not be None'
        assert mm_model is not None, 'mm_model can not be None'
        self.__model = mm_model
        engine_config = mm.Model.EngineConfig()
        engine_config.device_type = "MLU"
        self.__engine = self.__model.create_i_engine(engine_config)
        assert self.__engine is not None, 'Failed to create engine'
        self.__context = self.__engine.create_i_context()
        assert self.__context is not None, 'Failed to create context'
        self.__raw_input_tensors = self.__context.create_inputs()
        self.__queue = device.create_queue()

    def get_raw_input_tensors(self):
        return self.__raw_input_tensors

    def execute(self, input_tensors):
        output_tensors = []
        _check_status(self.__context.enqueue(input_tensors, output_tensors, self.__queue, input_by_tensor_order=True))
        _check_status(self.__queue.sync())
        return output_tensors

def load_model(device : mm.Device, path : str):
    ''' load model from disk '''
    model = mm.Model()
    assert model.deserialize_from_file(path).ok(), 'load model failed, model path: [%s]' % (path)
    return NNModel(device, model)

# create a new on-device mm.Tensor and fill its data with the input array
def ndarray2mlu(array):
    dtype_np2mm = {
        np.dtype('float32') : mm.DataType.FLOAT32,
        np.dtype('float16') : mm.DataType.FLOAT16,
        np.dtype('int32')   : mm.DataType.INT32
    }
    tensor = mm.Tensor(shape=array.shape, dtype=dtype_np2mm[array.dtype], location=mm.TensorLocation.kMLU)
    _check_status(tensor.from_numpy(array))
    return tensor

