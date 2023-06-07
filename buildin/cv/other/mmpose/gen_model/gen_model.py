import argparse
import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
from magicmind.python.runtime import DataType
from calibrator import CalibData
import time
import sys
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

import torch
import mmcv
from mmcv import Config, DictAction
from mmpose.datasets import build_dataloader, build_dataset

from logger import Logger

log = Logger()

def get_calibrate_data(_config,data_num):
    cfg = Config.fromfile(_config)
    
    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=False),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'seed',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }
    # step2: cfg.data.test_dataloader has higher priority
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **dict(workers_per_gpu=1),
        **dict(samples_per_gpu=1),
        **cfg.data.get('test_dataloader', {})
    }
    # print()
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    
    iter_idx = 0
    calib_data = []
    calib_finshed = False
    for data in data_loader:
        while not calib_finshed:
            img_tensor = data['img']
            assert img_tensor.size(0) == len(data['img_metas'].data[0])
            img_metas = data['img_metas'].data[0][0]
            aug_data = img_metas['aug_data']
            test_scale_factor = img_metas['test_scale_factor']
            for idx, s in enumerate(sorted(test_scale_factor, reverse=True)):
                if iter_idx < data_num:
                    image_resized = aug_data[idx].to(img_tensor.device)
                    np_data = image_resized[0].numpy()
                    calib_data.append(np_data)     
                else:
                    calib_finshed = True
                    break
                iter_idx += 1
    return calib_data

def get_network(args):
    parser = OnnxParser(args)
    network = parser.parse()
    return network

def calibrate(args, network: mm.Network, config: mm.BuilderConfig):
    # 创建量化工具并设置量化统计算法
    input_dims = mm.Dims(args.input_dims[0])
    sample_data = get_calibrate_data(args.config,args.batch_size[0])
    calib_data = CalibData(shape=input_dims, max_samples=args.batch_size[0], sample_data = sample_data)
    common_calibrate(args, network, config, calib_data)

def get_args():
    # get common argparser,here is pytorch_parser
    arg_parser = get_argparser()

    arg_parser.add_argument(
        "--config",
        dest="config",
        type=str,
        default="",
        help="config",
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
    model_name = "network"
    # build_and_serialize_params = extract_params("MODEL_BUILD_AND_SERIALIZE", args)
    build_and_serialize_args = extract_params("MODEL_BUILD_AND_SERIALIZE", args)
    build_and_serialize(network, build_config, build_and_serialize_args)

    end_time = time.time()
    log.info("gen_model time cost:{:.3f}s".format(end_time - begin_time))

if __name__ == "__main__":
    main()
