import argparse
import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
import time
import numpy as np

from preprocess import preprocess, imagenet_dataset
from calibrator import CalibData

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

log = Logger()

def get_network(args):
    parser = PytorchParser(args)
    network = parser.parse()
    output_count = network.get_output_count()
    return network

def get_args():
    # get common argparser,here is onnx_parser
    arg_parser = get_argparser()
    
    # add custom args belonging to the current net.
    arg_parser.add_argument(
        "--image_dir",
        dest="image_dir",
        type=str,
        default="",
        help="ImageNet",
    )
    arg_parser.add_argument(
        "--label_file ",
        dest="label_file",
        type=str,
        default="",
        help="label_file",
    )
    return arg_parser.parse_args()

def calibrate(args, network: mm.Network, config: mm.BuilderConfig):
    # 创建量化工具并设置量化统计算法
    dataset = imagenet_dataset(val_txt =args.label_file, image_file_path = args.image_dir, count=args.batch_size[0])
    sample_data = []
    for data, label in dataset:
        data = preprocess(data, transpose=False, normalization=True)
        sample_data.append(data)
    calib_data = CalibData([np.array(sample_data)])
    common_calibrate(args, network, config, calib_data)

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
    
    #临时加一下type64to32_conversion 等正式支持
    assert build_config.parse_from_string('{"opt_config":{"type64to32_conversion": true}}').ok()
    
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