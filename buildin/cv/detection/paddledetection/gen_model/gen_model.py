import magicmind.python.runtime as mm
from calibrator import CalibData, CalibDataShape, CalibDataScale,common_calibrate
import time
import numpy as np
# import common components
from framework_parser import OnnxParser
from build_param import get_argparser 
from model_process import (
    extract_params,
    config_network,
    get_builder_config,
    build_and_serialize,
)
from logger import Logger

log = Logger()

def get_network(args):
    parser = OnnxParser(args)
    network = parser.parse()
    return network


def calibrate(args, network: mm.Network, config: mm.BuilderConfig):
    # 创建量化工具并设置量化统计算法
    if args.model_name == "yolov3":
        input_dims = mm.Dims(args.input_dims[1])
        custom_max_samples = 10
        max_samples = max(custom_max_samples, args.input_dims[1][0])
        calib_data = CalibData(
            shape=input_dims, max_samples=max_samples, img_dir=args.image_dir
        )
        input_shape = CalibDataShape(max_samples, args.input_dims[1][2])
        scale_factor = CalibDataScale(max_samples)
        calib_data = [input_shape, calib_data, scale_factor]
        common_calibrate(args, network, config, calib_data)
    else : # ppyoloe
        input_dims = mm.Dims(args.input_dims[0])
        custom_max_samples = 1
        max_samples = max(custom_max_samples, args.input_dims[0][0])
        calib_data = CalibData(
            shape=input_dims, max_samples=max_samples, img_dir=args.image_dir
        )
        scale_factor = CalibDataScale(max_samples)
        calib_data = [calib_data, scale_factor]
        common_calibrate(args, network, config, calib_data)

def get_args():
    # get common argparser,here is onnx_parser
    arg_parser = get_argparser()

    # add custom args belonging to the current net.
    arg_parser.add_argument(
        "--image_dir",
        dest="image_dir",
        type=str,
        default="../../datasets/coco/val2017",
        help="coco2017 datasets",
    )
    arg_parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default="yolov3",
        help="yolov3 or ppyoloe",
    )
    return arg_parser.parse_args()


def main():
    begin_time = time.time()
    # get net args
    args = get_args()
    network = get_network(args)
    # configure network, such as input_dim, batch_size ...
    config_args = extract_params("MODEL_CONFIG", args)
    config_network(network, config_args)
    # create build configuration
    builder_args = extract_params("MODEL_BUILDER", args)
    build_config = get_builder_config(builder_args)

    if args.precision.find("qint") != -1:
        log.info("Do calibrate...")
        calibrate(args, network, build_config)
    log.info("Build model...")
    # 生成模型并导出
    build_and_serialize_args = extract_params("MODEL_BUILD_AND_SERIALIZE", args)
    build_and_serialize(network, build_config, build_and_serialize_args)

    end_time = time.time()
    log.info("gen_model time cost:{:.3f}s".format(end_time - begin_time))


if __name__ == "__main__":
    main()
