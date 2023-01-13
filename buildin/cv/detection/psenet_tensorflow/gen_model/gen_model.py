# -*- coding: utf-8 -*-
from magicmind.python.runtime.parser import Parser
import magicmind.python.runtime as mm
import cv2
import numpy as np
import os
import glob
import argparse

def do_calibrate(network, calib_data, config, precision):
    calibrator = mm.Calibrator([calib_data])
    assert calibrator is not None

    assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()
    assert config.parse_from_string(
        """{"precision_config": {"precision_mode": "%s"}}"""%(precision)).ok()
    # calibrate the network
    assert calibrator.calibrate(network, config).ok()

def preprocess_image(img, dst_shape) -> np.ndarray:

    img = cv2.resize(img, (dst_shape[0], dst_shape[1]))
    img = img.astype(dtype = np.float32)
    return img

class FixedCalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, max_samples: int, img_dir: str):
        super().__init__()
        assert os.path.isdir(img_dir)
        self.data_paths_ = glob.glob(img_dir + '/*.jpg')
        self.shape_ = shape
        self.max_samples_ = min(max_samples, len(self.data_paths_))
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        self.dst_shape_ = (self.shape_.GetDimValue(1), self.shape_.GetDimValue(2))

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
            img = preprocess_image(img, self.dst_shape_)
            imgs.append(img[np.newaxis,:])
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

def construct_model(args, reload_by_serilize=True):
    # init builder, network, builder_config and parser
    builder = mm.Builder()
    network = mm.Network()
    config = mm.BuilderConfig()
    parser = Parser(mm.ModelKind.kTensorflow)

    # get input dims from network
    parser.set_model_param("tf-model-type", "tf-graphdef-file")
    parser.set_model_param("tf-graphdef-inputs", ["input_images"])
    parser.set_model_param("tf-graphdef-outputs", ["Sigmoid"])
    
    assert parser.parse(network, args.tf_pb).ok()
    assert network.get_input(0).set_dimension(mm.Dims((args.batch_size, args.img_size[0], args.img_size[1], 3))).ok()
    config.parse_from_string('{"archs":[{"mtp_372": [2,6,8]}]}')
    config.parse_from_string('{"opt_config":{"type64to32_conversion": true}}')
    config.parse_from_string('{"opt_config":{"conv_scale_fold": true}}')
    if args.shape_mutable=='true':
        assert config.parse_from_string('{"graph_shape_mutable": true}').ok()
        assert config.parse_from_string('{"dim_range": {"0": {"min": [1, 704, 1216, 3], "max": [32, 704, 1216,3]}}}').ok()
    else:
        assert config.parse_from_string('{"graph_shape_mutable": false}').ok()
    
    # create calibrate data
    calib_data = FixedCalibData(mm.Dims([args.batch_size, args.img_size[0], args.img_size[1], 3]),
                                max_samples = args.batch_size,
                                img_dir = args.datasets_dir)
    
    do_calibrate(network, calib_data, config, args.precision)
    model = builder.build_model("quanmodel", network, config)
    assert model.serialize_to_file(args.mm_model).ok()
    print(args.mm_model," was saved successfully in the current path.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_pb', type=str, default='../data/models/psenet.pb', help='tf_pb path(s)')
    parser.add_argument('--mm_model', type=str, default='../data/', help='saved .mm model name')
    parser.add_argument('--precision', type=str, default='force_float32', help='precision')
    parser.add_argument('--shape_mutable', type=str, default="false", required=True ,help='shape_mutable')
    parser.add_argument('--datasets_dir', type=str, default='./icdar2015/images/', help='quantized data path,default: /data/datasets/COCO2017/images/val2017')
    parser.add_argument('--quan_img_num', type=int, default=5, help='quantized img num, default:10')
    parser.add_argument('--img_size', type=list, default=[704, 1216], help='inference size (pixels)')
    parser.add_argument('--batch_size', type=int, default=1, help='inference batch, default:1') 
    args = parser.parse_args()    

    construct_model(args)

    
