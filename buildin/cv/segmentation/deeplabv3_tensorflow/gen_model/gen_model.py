import os
import json
import numpy as np
import cv2
import argparse
import logging
import magicmind.python.runtime as mm
import magicmind.python.runtime.parser
from PIL import Image 
from calibrator import CalibData, Calibrator

parser = argparse.ArgumentParser()
parser.add_argument("--tf_model", type=str, default= "../data/models/frozen_inference_graph.pb", help="tf frozen pb model")
parser.add_argument("--image_dir",  type=str, default=os.path.join(str(os.environ.get("VOC2012_DATASETS_PATH")), "VOC2012","JPEGImages"), help="VOC2012 datasets")
parser.add_argument("--file_list", type=str, default=os.path.join(str(os.environ.get("VOC2012_DATASETS_PATH")), "VOC2012","ImageSets","Segmentation","val.txt"), help="val.txt path")
parser.add_argument("--output_model_path", type=str, default= "../data/models/deeplabv3_tensorflow_model_force_float32_true_1", help="save mm model to this path")
parser.add_argument("--precision", type=str, default="force_float32", help="qint8_mixed_float16, qint8_mixed_float32, qint16_mixed_float16, qint16_mixed_float32, force_float32, force_float16")
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

if __name__ == "__main__":
    args = parser.parse_args()
    input_size = 513
    batch_size = 1
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
    assert tf_parser.parse(deeplabv3_network, args.tf_model).ok()

    deeplabv3_network.get_input(0).set_dimension(mm.Dims([batch_size, input_size, input_size, 3]))
    # 指定硬件平台
    assert config.parse_from_string('{"archs":[{"mtp_372": [2,6,8]}]}').ok()
    # INT64 转 INT32
    assert config.parse_from_string('{"opt_config":{"type64to32_conversion":true}}').ok()
    # conv_scale_fold
    assert config.parse_from_string('{"opt_config":{"conv_scale_fold":true}}').ok()
    # 模型输入输出规模可变功能
    assert config.parse_from_string('{"graph_shape_mutable":false}').ok()
    # 精度模式
    assert config.parse_from_string('{"precision_config":{"precision_mode":"%s"}}' % args.precision).ok()

    dataset = voc_dataset(file_list = args.file_list, image_file_path = args.image_dir, count = batch_size)
    sample_data = []
    for data in dataset:
        width, height = data.size
        resize_ratio = 1.0 * input_size / max(width, height)
        target_size = (input_size, input_size)
        resized_image = data.convert('RGB').resize(target_size, Image.ANTIALIAS)
        resized_image = np.asarray(resized_image)
        input_data = np.copy(resized_image)
        sample_data.append(np.expand_dims(input_data, 0))

    if "qint" in args.precision:
        calib_data = CalibData(sample_data)
        calibrator = Calibrator(calib_data, mm.QuantizationAlgorithm.LINEAR_ALGORITHM)
        assert config.parse_from_string('{"precision_config": {"weight_quant_granularity": "per_tensor"}}').ok()
        assert calibrator.calibrate(deeplabv3_network, config).ok()

    builder = mm.Builder()
    model = builder.build_model("deeplabv3_mm_model", deeplabv3_network, config)
    assert model != None
    model.serialize_to_file(args.output_model_path)
    logging.info("Generate model done, model save to %s" % args.output_model_path) 
