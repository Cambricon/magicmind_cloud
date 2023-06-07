import os
import time 
import torch 
import numpy as np 
import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
import argparse
import sys
import glob
sys.path.append('../')
from infer_python.preprocess import preprocess
import cv2

class CalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, max_samples: int, img_dir: str):
        super().__init__()
        assert os.path.isdir(img_dir)
        self.data_paths_ = glob.glob(img_dir + '/*.png')
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
            img, _ = preprocess(img)
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


def generate_config(args):
    config = mm.BuilderConfig()
    # 指定硬件平台
    assert config.parse_from_string('{"archs":[{"mtp_372": [2,6,8]}]}').ok()
    # 优化项
    assert config.parse_from_string('{"opt_config":{"type64to32_conversion": true}}').ok()
    assert config.parse_from_string('{"opt_config":{"conv_scale_fold": true}}').ok()
    assert config.parse_from_string('{"computation_preference": "fast"}').ok
    assert config.parse_from_string('{"debug_config":{"fusion_enable":false}}').ok()
    # 模型输入输出规模可变功能
    assert config.parse_from_string('{      \
        "graph_shape_mutable": true,        \
        "dim_range": {                      \
        "0": {                              \
            "min": [1, 3, 1024, 2048],      \
            "max": [16,3, 1024, 2048]       \
        }                                   \
    }}').ok()
    # 精度模式
    assert config.parse_from_string('{"precision_config":{"precision_mode":"%s"}}' % args.precision).ok()
    return config

def calibrate(args, network : mm.Network, config : mm.BuilderConfig):
    # 创建量化工具并设置量化统计算法
    calib_data = CalibData(shape = mm.Dims([args.batch_size, 3, args.input_height, args.input_width]), 
                           max_samples = args.batch_size, 
                           img_dir = args.image_dir)
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
        print("Working on MLU:", args.device_id)
    # 进行量化
    assert calibrator.calibrate(network, config).ok()


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--onnx_model', dest = 'onnx_model',
            required = True, type = str, help = 'onnx model path')
    args.add_argument('--output_model', dest = 'output_model', default = '',
            type = str, help = 'output model path')
    args.add_argument('--precision', dest = 'precision', default = 'force_float32',
            type = str, help = 'precision, only qint8_mixed_float16 qint8_mixed_float32 force_float16 force_float32 qint16_mixed_float32 are supported')
    args.add_argument('--image_dir', dest = 'image_dir', default = 'image_dir',
            type = str, help = 'image list file path, file contains input image paths for calibration')
    args.add_argument('--batch_size', dest = 'batch_size', default = 1,
            type = int, help = 'batch_size')
    args.add_argument('--input_width', dest = 'input_width', default = 2048,
            type = int, help = 'model input width')
    args.add_argument('--input_height', dest = 'input_height', default = 1024,
            type = int, help = 'model input height')
    args.add_argument('--device_id', dest = 'device_id', default = 0,
            type = int, help = 'mlu device id, used for calibration')
    args = args.parse_args()
    print(args)
    supported_precision = ['qint8_mixed_float16', 'qint8_mixed_float32', 'qint16_mixed_float16', 'qint16_mixed_float32', 'force_float16', 'force_float32']
    if args.precision not in supported_precision:
        print('precision [' + args.precision + ']', 'not supported')
        exit()

    # 生成mm模型
    parser = Parser(mm.ModelKind.kOnnx)
    network = mm.Network()
    assert parser.parse(network, args.onnx_model).ok()
    build_config = generate_config(args)

    if args.precision.find('qint') != -1:
        print('do calibrate...')
        calibrate(args, network, build_config)
    print('build model...')

    builder = mm.Builder()
    model = builder.build_model("ocrnet", network, build_config)
    if '' == args.output_model:
        args.output_model = 'ocrnet_{}.mm'.format(args.precision)
    assert model.serialize_to_file(args.output_model).ok()
    print("Generate model done, model save to %s" % args.output_model) 

if __name__=='__main__':
    main()
