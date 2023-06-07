import os
import numpy as np
import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
import argparse
from calibrator import CalibData
from typing import List

import time

# import common components
from framework_parser import CaffeParser
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
    parser = CaffeParser(args)
    network = parser.parse()
    return network

def calibrate(args, network: mm.Network, config: mm.BuilderConfig):
    input_dims = mm.Dims(args.input_dims[0])
    custom_max_samples = 1
    max_samples = max(custom_max_samples, args.input_dims[0][0])
    calib_data = CalibData(
        shape=input_dims,
        max_samples=max_samples,
        img_dir=args.image_dir,
        need_insert_bn = True,
        mean=args.means,
        std=args.vars,
    )
    common_calibrate(args, network, config, calib_data)

def get_args():
    # get common argparser,here is pytorch_parser
    arg_parser = get_argparser()

    # add custom args belonging to the current net.
    arg_parser.add_argument(
        "--image_dir",
        dest="image_dir",
        type=str,
        default="../../datasets/ilsvrc2012",
        help="ilsvrc2012 datasets",
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
    # build model and export the model to disk
    model_name = "network"
    build_and_serialize_args = extract_params("MODEL_BUILD_AND_SERIALIZE", args)
    build_and_serialize(network, build_config, build_and_serialize_args)

    end_time = time.time()
    log.info("gen_model time cost:{:.3f}s".format(end_time - begin_time))


if __name__ == "__main__":
    main()

