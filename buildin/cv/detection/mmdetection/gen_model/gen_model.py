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
from logger import Logger

log = Logger()

# mmdetection datasets
sys.path.append('../export_model/mmdetection')
from mmcv import Config, DictAction
from mmdet.utils import compat_cfg
from mmdet.datasets import ( build_dataloader, build_dataset,replace_ImageToTensor )

def is_contain(node_name:str, ops:list):
    for op in ops:
        if node_name.__contains__(op):
            return True
    return False

def get_mmdetecion_data(args,max_samples):
    cfg = Config.fromfile(args.config)
    cfg = compat_cfg(cfg)
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu= 1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False)
    
    dataset = data_loader.dataset
    sample_idx = 0
    sample_data = []
    for i, data in enumerate(data_loader):
        if sample_idx > max_samples:
            break
        img = data['img'][0].numpy()[0]
        sample_data.append(img)
        sample_idx += 1
    return sample_data

def get_network(args):
    parser = OnnxParser(args)
    network = parser.parse()
    node_list =  network.get_all_nodes_in_network()
    node_types = []
    op_list=['NonMaxSuppression_', 'Sqrt_', 'Mul_'] 

    for _node in node_list:
        node_name = _node.get_node_name()
        node_type = _node.get_node_type()
        node_types.append(node_type)
        if is_contain(node_name,op_list):
            input_count = _node.get_input_count()
            output_count = _node.get_output_count()
            
            # check input
            for _input_idx  in range(input_count):
                _input = _node.get_input(_input_idx)
                _input_dtype = _input.get_data_type()
                if _input_dtype in [ mm.DataType.FLOAT16, mm.DataType.FLOAT32 ]:
                    _node.set_precision(_input_idx,mm.DataType.FLOAT32)
                    
            # check output            
            for _output_idx  in range(output_count):
                _output = _node.get_output(_output_idx)
                _output_dtype = _output.get_data_type()
                if _output_dtype in [ mm.DataType.FLOAT16, mm.DataType.FLOAT32 ]:
                    _node.set_output_type(_output_idx,mm.DataType.FLOAT32)
    return network

def calibrate(args, network: mm.Network, config: mm.BuilderConfig):
    # 创建量化工具并设置量化统计算法
    input_dims = mm.Dims(args.input_dims[0])
    max_samples = args.batch_size[0]
    sample_data = get_mmdetecion_data(args,max_samples)
    
    calib_data = CalibData(shape=input_dims, max_samples=max_samples, sample_data = sample_data)
    common_calibrate(args, network, config, calib_data)

def get_args():
    # get common argparser,here is pytorch_parser
    arg_parser = get_argparser()

    # add custom args belonging to the current net.
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
