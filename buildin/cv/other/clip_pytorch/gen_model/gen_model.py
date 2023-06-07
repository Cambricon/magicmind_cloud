import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
import numpy as np
import glob
import os 
import argparse
import cv2
from calibrator import CalibData
import time

# import common components
from framework_parser import OnnxParser
from common_calibrate import common_calibrate
from build_param import get_argparser
from model_process import (
    extract_params,
    config_network,
    get_builder_config,
    build_and_serialize,
)
from logger import Logger

log = Logger()

WIDTH = 224
HEIGHT = 224
MAX_SAMPLES = 1000

def get_network(args):
    parser = OnnxParser(args)
    network = parser.parse()
    output_count = network.get_output_count()
    return network

def get_args():
    # get common argparser,here is onnx_parser
    arg_parser = get_argparser()
    arg_parser.add_argument(
        "--image_dir",
        dest="image_dir",
        type=str,
        default= "../../datasets/cifar100",
        help="image_dir",
    )

    # add custom args belonging to the current net.
    return arg_parser.parse_args()

def calibrate(args, network: mm.Network, config: mm.BuilderConfig):
    # 创建量化工具并设置量化统计算法
    sample_data = []
    #dim = mm.Dims((args.batch_size[0], 3, HEIGHT, WIDTH))
    #data1 = CalibData(dim,0, MAX_SAMPLES)
    sample_data.append(CalibData(mm.Dims((args.batch_size[0], 3, HEIGHT, WIDTH)), 0, MAX_SAMPLES))
    sample_data.append(CalibData(mm.Dims((args.batch_size[0], 3, HEIGHT, WIDTH)), 1, MAX_SAMPLES))
    #common_calibrate(args, network, config, sample_data)
    calibrator = mm.Calibrator(sample_data)
    #assert calibrator is not None
    # 设置量化统计算法，支持线性统计算法（LINEAR_ALGORITHM）及加强的最小化量化噪声算法（EQM_ALGORITHM）。
    assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()
    assert calibrator.calibrate(network, config).ok()


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
    #build_and_serialize_params = extract_params("MODEL_BUILD_AND_SERIALIZE", args)
    build_and_serialize_args = extract_params("MODEL_BUILD_AND_SERIALIZE", args)
    build_and_serialize(network, build_config, build_and_serialize_args)

    end_time = time.time()
    log.info("gen_model time cost:{:.3f}s".format(end_time - begin_time))

if __name__ == "__main__":
    main()

