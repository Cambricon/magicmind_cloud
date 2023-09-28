import argparse
import os
import numpy as np
from skimage import transform, io
import magicmind.python.runtime as mm
from magicmind.python.runtime import DataType
from magicmind.python.common.types import get_numpy_dtype_by_datatype
import cv2
from magicmind.python.runtime.parser import Parser

MSRA_B_DATASETS_PATH = os.getenv("MSRA_B_DATASETS_PATH")

def torch_parser(args):
    # 创建MagicMind parser
    parser = Parser(mm.ModelKind.kPytorch)
    # 设置网络输入数据类型
    parser.set_model_param("pytorch-input-dtypes", [mm.DataType.FLOAT32])
    # 创建一个空的网络实例
    network = mm.Network()
    # 使用parser将PyTorch模型文件转换为MagicMind Network实例。
    assert parser.parse(network, args.pt_model).ok()
    # 设置模型输入形状
    input_dims = mm.Dims((args.batch_size, 3, args.input_height, args.input_width))
    assert network.get_input(0).set_dimension(input_dims).ok()
    return network

def generate_model_config(args):
    config = mm.BuilderConfig()
    # 指定硬件平台
    assert config.parse_from_string('{"archs":[{"mtp_372": [6,8]}]}').ok()
    assert config.parse_from_string('{"opt_config":{"conv_scale_fold":true}}').ok()
    assert config.parse_from_string('{"precision_config":{"precision_mode":"%s"}}' % args.precision).ok()
    return config

def preprocess(img):
    output_size = (320, 320)
    new_h, new_w = output_size
    image = transform.resize(img,(new_h, new_w),mode='constant')
    tmpImg = np.zeros((image.shape[0], image.shape[1],3))
    image = image/np.max(image)
    if image.shape[2] == 1:
        tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
        tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
        tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
    else:
        tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
        tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
        tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225
    tmpImg = tmpImg.transpose((2, 0, 1))
    return tmpImg

def load_processed_image(file_path):
    image = cv2.imread(file_path)
    tmpImg = preprocess(image)
    return tmpImg

def load_multi_image(data_paths, target_dtype: mm.DataType = mm.DataType.FLOAT32) -> np.ndarray:
    # Load multiple pre-processed image into a NCHW style ndarray
    images = []
    with open(data_paths[0]) as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = MSRA_B_DATASETS_PATH + os.sep + line
            images.append(load_processed_image(line)[np.newaxis, :])
    ret = np.concatenate(tuple(images))
    return np.ascontiguousarray(ret.astype(dtype=get_numpy_dtype_by_datatype(target_dtype)))

class FixedCalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, data_type: mm.DataType, max_samples: int,
                 data_paths):
        super().__init__()
        self.shape_ = shape
        self.data_type_ = data_type
        self.batch_size_ = shape.GetDimValue(0)
        self.max_samples_ = min(max_samples, len(data_paths))
        self.data_paths_ = data_paths
        self.current_sample_ = None
        self.outputed_sample_count = 0

    def get_shape(self):
        return self.shape_

    def get_data_type(self):
        return self.data_type_

    def get_sample(self):
        return self.current_sample_

    def next(self):
        beg_ind = self.outputed_sample_count
        end_ind = self.outputed_sample_count + self.max_samples_
        if end_ind > self.max_samples_:
            return mm.Status(mm.Code.OUT_OF_RANGE, "End reached")

        self.current_sample_ = load_multi_image(self.data_paths_, 
                                                target_dtype=self.data_type_)
        self.outputed_sample_count = end_ind
        return mm.Status.OK()

    def reset(self):
        self.current_sample_ = None
        self.outputed_sample_count = 0
        return mm.Status.OK()

def calibrate(args, network : mm.Network, config : mm.BuilderConfig):
    # 创建量化工具并设置量化统计算法
    calib_data = FixedCalibData(shape = mm.Dims([args.batch_size, 3, args.input_height, args.input_width]),
                                data_type = DataType.FLOAT32,
                                max_samples = 10,
                                data_paths = [args.file_list])
    calibrator = mm.Calibrator([calib_data])
    assert calibrator is not None
    # 设置量化统计算法，支持线性统计算法（LINEAR_ALGORITHM）及加强的最小化量化噪声算法（EQM_ALGORITHM）。
    assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()
    # 打开设备
    with mm.System() as mm_sys:
        dev_count = mm_sys.device_count()
        print("Device count: ", dev_count)
        if args.device_id >= dev_count:
            print("Invalid device set!")
            abort()
        # 打开MLU设备
        dev = mm.Device()
        dev.id = args.device_id
        assert dev.active().ok()
    # 进行量化
    assert calibrator.calibrate(network, config).ok()

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--pt_model', dest = 'pt_model', default = '/workspace/saved_pt/2d_unet_0.pt',
            required = True, type = str, help = 'tf output graph')
    args.add_argument('--output_model', dest = 'output_model', default = '/workspace/offline_models/2d_unet_0.model',
            type = str, help = 'output model path')
    args.add_argument('--precision', dest = 'precision', default = 'qint8_mixed_float16',
            type = str, help = 'precision, qint8_mixed_float16 qint8_mixed_float32 force_float16 force float32 qint16_mixed_float32 are supported')
    args.add_argument('--file_list', dest = 'file_list', default = 'file_list',
            type = str, help = 'image list file path, file contains input image paths for calibration')
    args.add_argument('--batch_size', dest = 'batch_size', default = 1,
            type = int, help = 'batch_size')
    args.add_argument('--input_width', dest = 'input_width', default = 320,
            type = int, help = 'model input width')
    args.add_argument('--input_height', dest = 'input_height', default = 320,
            type = int, help = 'model input height')
    args.add_argument('--device_id', dest = 'device_id', default = 0,
            type = int, help = 'mlu device id, used for calibration')
    args = args.parse_args()

    supported_precision = ['qint8_mixed_float16', 'qint8_mixed_float32', 'force_float16', 'force_float32', 'qint16_mixed_float32']
    if args.precision not in supported_precision:
        print('precision [' + args.precision + ']', 'not supported')
        exit()

    network = torch_parser(args)
    config = generate_model_config(args)
    if args.precision.find('qint') != -1:
        print('do calibrate...')
        calibrate(args, network, config)
    print('build model...')
    # 生成模型
    builder = mm.Builder()
    model = builder.build_model('magicmind model', network, config)
    assert model is not None
    # 将模型序列化为离线文件
    assert model.serialize_to_file(args.output_model).ok()
    print("Generate model done, model save to %s" % args.output_model)

if __name__ == "__main__":
    main()
