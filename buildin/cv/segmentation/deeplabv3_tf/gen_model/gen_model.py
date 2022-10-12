import os
import json
import numpy as np
import cv2
import argparse
import logging
import magicmind.python.runtime as mm
import magicmind.python.runtime.parser
from magicmind.python.common.types import get_datatype_by_numpy
from typing import List
from PIL import Image 

parser = argparse.ArgumentParser()
parser.add_argument("--tf_model", type=str, default= "../data/models/frozen_inference_graph.pb", help="tf frozen pb model")
parser.add_argument("--image_dir",  type=str, default=os.path.join(str(os.environ.get("DATASETS_PATH")), "VOC2012","JPEGImages"), help="VOC2012 datasets")
parser.add_argument("--file_list", type=str, default=os.path.join(str(os.environ.get("DATASETS_PATH")), "VOC2012","ImageSets","Segmentation","val.txt"), help="val.txt path")
parser.add_argument("--output_model_path", type=str, default= "../data/models/deeplabv3_tf_model_force_float32_true_1", help="save mm model to this path")
parser.add_argument("--quant_mode", type=str, default="force_float32", help="qint8_mixed_float16, qint8_mixed_float32, qint16_mixed_float16, qint16_mixed_float32, force_float32, force_float16")
parser.add_argument("--shape_mutable", type=str, default="true", help="whether the mm model is dynamic or static or not")

def voc_dataset(file_list, image_file_path, count):
    with open(file_list, "r") as f:
        lines = f.readlines()
    logging.info("%d pictures will be read." % len(lines))
    current_count = 0
    for line in lines:
        image_name = line.replace("\n", "")
        image_path = os.path.join(image_file_path, image_name + ".jpg")
        img = Image.open(image_path)
        yield img
        current_count += 1
        if current_count > count and count != -1:
            break

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


class Calibrator(mm.Calibrator):
    def __init__(self, calibdata, algorithm):
        super(Calibrator, self).__init__(calibdata)
        status = self.set_quantization_algorithm(algorithm)
        assert status.ok(), str(status)

    def calibrate(self, network, builder_config=None):
        status = super(Calibrator, self).calibrate(network, builder_config)
        if not status.ok():
            return status
        return mm.Status()

if __name__ == "__main__":
    args = parser.parse_args()

    tf_model = args.tf_model
    file_list = args.file_list
    image_file_path = args.image_dir
    batch_size = 1
    input_size = 513
    shape_mutable = args.shape_mutable
    offline_model_name = args.output_model_path
    # 创建 MagicMind Network
    deeplabv3_network = mm.Network()
     # 创建 MagicMind Config
    config = mm.BuilderConfig()
    # 创建 MagicMind Parser （tensorflow后端）
    tf_parser = mm.parser.Parser(mm.ModelKind.kTensorflow)
    tf_parser.set_model_param("tf-model-type", "tf-graphdef-file")
    tf_parser.set_model_param("tf-graphdef-inputs",["ImageTensor:0"])
    tf_parser.set_model_param("tf-graphdef-outputs", ["SemanticPredictions:0"])
    tf_parser.set_model_param("tf-infer-shape", True)
    assert tf_parser.parse(deeplabv3_network, tf_model).ok()

    deeplabv3_network.get_input(0).set_dimension(mm.Dims([batch_size, input_size, input_size, 3]))
    # 指定硬件平台
    assert config.parse_from_string('{"archs":[{"mtp_372": [6,8]}]}').ok()
    # INT64 转 INT32
    assert config.parse_from_string('{"opt_config":{"type64to32_conversion":true}}').ok()
    # conv_scale_fold
    assert config.parse_from_string('{"opt_config":{"conv_scale_fold":true}}').ok()
    # 模型输入输出规模可变功能
    if args.shape_mutable == "true":
        assert config.parse_from_string('{"graph_shape_mutable":true}').ok()
        assert config.parse_from_string('{"dim_range": {"0": {"min": [1, 100, 100, 3], "max": [1, 513, 513, 3]}}}').ok()
    else:
        assert config.parse_from_string('{"graph_shape_mutable":false}').ok()

    ###quantazation mode
    if args.quant_mode == "qint8_mixed_float16":
        assert config.parse_from_string('{"precision_config": {"precision_mode": "qint8_mixed_float16"}}').ok()
    elif args.quant_mode == "qint8_mixed_float32":
        assert config.parse_from_string('{"precision_config": {"precision_mode": "qint8_mixed_float32"}}').ok()
    elif args.quant_mode == "qint16_mixed_float16":
        assert config.parse_from_string('{"precision_config": {"precision_mode": "qint16_mixed_float16"}}').ok()
    elif args.quant_mode == "qint16_mixed_float32":
        assert config.parse_from_string('{"precision_config": {"precision_mode": "qint16_mixed_float32"}}').ok()
    elif args.quant_mode == "force_float32": 
        assert config.parse_from_string('{"precision_config": {"precision_mode": "force_float32"}}').ok()
    elif args.quant_mode == "force_float16": 
        assert config.parse_from_string('{"precision_config": {"precision_mode": "force_float16"}}').ok()

    dataset = voc_dataset(file_list = file_list, image_file_path = image_file_path, count = batch_size)
    sample_data = []
    for data in dataset:
        width, height = data.size
        resize_ratio = 1.0 * input_size / max(width, height)
        if args.shape_mutable == "true":
          target_size = (int(resize_ratio * width), int(resize_ratio * height))
        if args.shape_mutable == "false":
          target_size = (input_size, input_size)
        resized_image = data.convert('RGB').resize(target_size, Image.ANTIALIAS)
        resized_image = np.asarray(resized_image)
        input_data = np.copy(resized_image)
        sample_data.append(np.expand_dims(input_data, 0))

    if "qint" in args.quant_mode:
        calib_data = CalibData(sample_data)
        calibrator = Calibrator(calib_data, mm.QuantizationAlgorithm.LINEAR_ALGORITHM)
        assert config.parse_from_string('{"precision_config": {"weight_quant_granularity": "per_tensor"}}').ok()
        assert calibrator.calibrate(deeplabv3_network, config).ok()

    builder = mm.Builder()
    model = builder.build_model("deeplabv3_mm_model", deeplabv3_network, config)
    assert model != None
    model.serialize_to_file(offline_model_name)
    logging.info("Generate model done, model save to %s" % offline_model_name) 
