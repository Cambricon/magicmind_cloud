import argparse
import magicmind.python.runtime as mm
from calibrator import CalibData
from preprocess import preprocess, imagenet_dataset
import time
import numpy as np

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

def get_network(args):
    parser = OnnxParser(args)
    network = parser.parse()
    return network

def calibrate(args, network : mm.Network, config : mm.BuilderConfig):
    # 创建量化工具并设置量化统计算法
    batch_size = args.input_dims[0][0]
    dst_size = (args.input_dims[0][2],args.input_dims[0][3])
    dataset = imagenet_dataset(val_txt =args.label_file, image_file_path = args.image_dir, count=batch_size)
    sample_data = []
    for data, label in dataset:
        data = preprocess(data, dst_size, transpose=True)
        sample_data.append(data)
    calib_data = CalibData([np.array(sample_data)])
    common_calibrate(args, network, config, calib_data)  

def get_args():
    # get common argparser,here is pytorch_parser
    arg_parser = get_argparser()

    # add custom args belonging to the current net.
    arg_parser.add_argument(
        "--image_dir",
        dest="image_dir",
        type=str,
        default="ILSVRC2012_DATASETS_PATH",
        help="ILSVRC2012_DATASETS_PATH",
    )
    arg_parser.add_argument(
        "--label_file", "--label_file",  
        type=str, default="/path/to/modelzoo/datasets/imageNet2012/labels.txt",
        help="imagenet val label txt")
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

