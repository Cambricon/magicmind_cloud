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

def get_network(args):
    parser = OnnxParser(args)
    network = parser.parse()
    return network

def get_args():
    # get common argparser,here is onnx_parser
    arg_parser = get_argparser()
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
    assert build_config.parse_from_string('{"debug_config": {"fusion_enable": false}}').ok()

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

