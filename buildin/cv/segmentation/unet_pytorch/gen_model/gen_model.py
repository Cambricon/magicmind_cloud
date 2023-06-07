import argparse
import numpy as np
import magicmind.python.runtime as mm
import cv2
import json
import os 
#from calibrator import FixedCalibData
from magicmind.python.runtime.parser import Parser
from calibrator import CalibData
import time

# import common components
from framework_parser import PytorchParser
from common_calibrate import common_calibrate
from build_param import get_argparser
from model_process import (
    extract_params,
    config_network,
    get_builder_config,
    build_and_serialize,
)
from logger import Logger
# 实例化python logger类
log = Logger()


MODEL_PATH = os.getenv("MODEL_PATH")
CARVANA_DATASETS_PATH = os.getenv("CARVANA_DATASETS_PATH")

def get_network(args):
    parser = PytorchParser(args)
    network = parser.parse()
    #input_dims = args.input_dims
    #assert network.get_input(0).set_dimension(input_dims).ok()
    output_count = network.get_output_count()
    return network

def calibrate(args, network: mm.Network, config: mm.BuilderConfig):
    # 创建量化工具并设置量化统计算法
    input_dims = mm.Dims(args.input_dims[0])
    custom_max_samples = 50
    max_samples = max(custom_max_samples, args.input_dims[0][0])
    calib_data = CalibData(
        shape=input_dims, scale=args.scale, max_samples=max_samples, img_dir=args.image_dir
    )
    common_calibrate(args, network, config, calib_data)

def get_args():
    # get common argparser,here is pytorch_parser
    #_, arg_parser, _,_ = parse_build_params()
    arg_parser = get_argparser()
    
    arg_parser.add_argument(
        "--image_dir",
        dest="image_dir",
        type=str,
        default= str(CARVANA_DATASETS_PATH) + '/imgs/',
        help="image_dir",
    )
    arg_parser.add_argument(
        "--scale",
        dest="scale",
        type=float,
        default=0.5,
        help="scale",
    )

    # add custom args belonging to the current net.
    return arg_parser.parse_args()

def main():
    # get net args
    begin_time = time.time()
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
    model_name = "network"
    # build_and_serialize_params = extract_params("MODEL_BUILD_AND_SERIALIZE", args)
    build_and_serialize_args = extract_params("MODEL_BUILD_AND_SERIALIZE", args)
    build_and_serialize(network, build_config, build_and_serialize_args)
    end_time = time.time()
    log.info("gen_model time cost:{:.3f}s".format(end_time - begin_time))
if __name__ == "__main__":
    main()
