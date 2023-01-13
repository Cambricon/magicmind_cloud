import argparse
import os
import numpy as np
import magicmind.python.runtime as mm
import cv2

DATASETS_PATH = os.environ.get("DATASETS_PATH")

def caffe_parser(args):
    from magicmind.python.runtime.parser import Parser
    network = mm.Network()
    parser = Parser(mm.ModelKind.kCaffe)
    assert parser.parse(network, args.caffemodel, args.prototxt).ok()
    input_dims = mm.Dims((args.batchsize, 3, args.input_height, args.input_width))
    assert network.get_input(0).set_dimension(input_dims).ok()
    return network

def generate_model_config(args):
    config = mm.BuilderConfig()
    assert config.parse_from_string('{"precision_config":{"precision_mode":"%s"}}' % args.precision).ok()
    assert config.parse_from_string('{"opt_config":{"type64to32_conversion":true}}').ok()
    assert config.parse_from_string('{"opt_config":{"conv_scale_fold":true}}').ok()
    # assert config.parse_from_string('{"opt_config":{"tfu_enable":false}}').ok()
    assert config.parse_from_string('{"graph_shape_mutable":false}').ok()
    # 量化算法，支持对称量化（symmetric)和非对称量化（asymmetric）。当量化统计算法设置为EQNM_ALOGORITHM时，仅适用于对称量化。
    assert config.parse_from_string('{"precision_config": {"activation_quant_algo": "symmetric"}}').ok()
    # 设置量化粒度，支持按tensor量化（per_tensor）和按通道量化（per_axis）两种。
    assert config.parse_from_string('{"precision_config": {"weight_quant_granularity": "per_tensor"}}').ok()
    # 指定硬件平台
    assert config.parse_from_string('{"archs":["mtp_372"]}').ok()
    # 将网络输入数据摆放顺序由NCHW转为NHWC
    assert config.parse_from_string('{"convert_input_layout":{"0":{"src":"NCHW", "dst":"NHWC"}}}').ok()
    # 将预处理中标准化过程集成到模型中(img = (img - mean) / std), 其中var的值为std的平方
    assert config.parse_from_string('{"insert_bn_before_firstnode":{"0":{"mean":[128,128,128],"var":[65536,65536,65536]}}}').ok()
    return config

def calibrate(args, network : mm.Network, config : mm.BuilderConfig):
    class MMCalibData(mm.CalibDataInterface):
        def __init__(self, args):
            super().__init__()
            with open(args.calibrate_list, 'r') as f:
                image_paths = f.readlines()

            self.images = []
            for image_path in image_paths:
                image = cv2.imread(str(DATASETS_PATH) + '/' + image_path.strip())
                assert image is not None, 'image [' + image_path.strip() + '] not exists!'
                self.images.append(image)
            nimages = len(self.images)
            assert nimages != 0, 'no images in calibrate list[' + args.calibrate_list + ']!'
            # at least one batch
            if nimages < args.batchsize:
                for i in range(args.batchsize - nimages):
                    self.images.append(self.images[0])
            self.shape_ = mm.Dims((args.batchsize, 3, args.input_height, args.input_width))
            self.cur_image_index_ = 0
    
        def get_shape(self):
            return self.shape_
    
        def get_data_type(self):
            return mm.DataType.FLOAT32
    
        def get_sample(self):
            return self.cur_sample_
    
        def preprocess_image(self):
            if self.cur_image_index_ == len(self.images):
                return None
            h = self.shape_.GetDimValue(2)
            w = self.shape_.GetDimValue(3)
            image = self.images[self.cur_image_index_]
            scaling_factor = min((h - 1) / (image.shape[0] - 1), (w - 1) / (image.shape[1] - 1))
            m = np.zeros((2, 3))
            m.astype('float64')
            m[0, 0] = scaling_factor
            m[1, 1] = scaling_factor
            image = cv2.warpAffine(image, m, (w, h),
                    flags = cv2.INTER_CUBIC if scaling_factor > 1.0 else cv2.INTER_AREA,
                    borderMode = cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            image = image.astype('float32')
            image -= 128
            image /= 256.0
            image = np.transpose(image, (2, 0, 1)) # HWC >>> CHW
            self.cur_image_index_ = self.cur_image_index_ + 1
            return image
    
        def next(self):
            batch_size = self.shape_.GetDimValue(0)
            preprocessed_images = []
            for i in range(batch_size):
                image = self.preprocess_image()
                if image is None:
                    # no more data
                    return mm.Status(mm.Code.OUT_OF_RANGE, "Data end reached")
                preprocessed_images.append(image)
            self.cur_sample_ = np.array(preprocessed_images)
            return mm.Status.OK()
    
        def reset(self):
            self.cur_sample_ = None
            self.cur_image_index_ = 0
            return mm.Status.OK()

    calib_data = MMCalibData(args)
    calibrator = mm.Calibrator([calib_data])
    assert calibrator is not None
    # 设置量化统计算法，支持线性统计算法（LINEAR_ALGORITHM）及加强的最小化量化噪声算法（EQM_ALGORITHM）。
    assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()
    # 打开设备
    with mm.System() as mm_sys:
        dev_count = mm_sys.device_count()
        print("Device count: ", dev_count)
        if args.device >= dev_count:
            print("Invalid device set!")
        # 打开MLU设备
        dev = mm.Device()
        dev.id = args.device
        assert dev.active().ok()
        # 进行量化
        assert calibrator.calibrate(network, config).ok()

def main():
    args = argparse.ArgumentParser(description='openpose caffe model to magicmind model')
    args.add_argument('--prototxt', dest = 'prototxt', default = 'pose_deploy.prototxt',
            required = True, type = str, help = 'prototxt file path')
    args.add_argument('--caffemodel', dest = 'caffemodel', default = 'pose_iter_584000.caffemodel',
            required = True, type = str, help = 'caffemodel file path')
    args.add_argument('--batchsize', dest = 'batchsize', default = 1,
            type = int, help = 'batchsize')
    args.add_argument('--input_width', dest = 'input_width', default = 656,
            type = int, help = 'model input width')
    args.add_argument('--input_height', dest = 'input_height', default = 368,
            type = int, help = 'model input height')
    args.add_argument('--output_model', dest = 'output_model', default = 'pose.magicmind',
            type = str, help = 'output model path')
    args.add_argument('--precision', dest = 'precision', default = 'qint8_mixed_float16',
            type = str, help = 'precision mode, qint8_mixed_float16 qint8_mixed_float32 force_float16 force float32 are supported')
    args.add_argument('--calibrate_list', dest = 'calibrate_list', default = 'calibrate_list.txt',
            type = str, help = 'image list file path, file contains input image paths for calibration')
    args.add_argument('--device', dest = 'device', default = 0,
            type = int, help = 'mlu device id, used for calibration')
    args = args.parse_args()

    supported_precision = ['qint8_mixed_float16', 'qint8_mixed_float32', 'force_float16', 'force_float32']
    if args.precision not in supported_precision:
        print('precision mode [' + args.precision + ']', 'not supported')
        exit()

    network = caffe_parser(args)
    config = generate_model_config(args)
    if args.precision.find('qint') != -1:
        print('do calibrate...')
        calibrate(args, network, config)
    print('build model...')
    builder = mm.Builder()
    model = builder.build_model('magicmind model', network, config)
    assert model is not None
    assert model.serialize_to_file(args.output_model).ok()
    print(args.output_model + " generate ok")

if __name__ == "__main__":
    main()

