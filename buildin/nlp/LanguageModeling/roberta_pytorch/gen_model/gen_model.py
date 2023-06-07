import argparse
import magicmind.python.runtime as mm
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
    # set input shape and data type
    input_dims = mm.Dims((args.input_dims[0]))
    for i in range(network.get_input_count()):
        network.get_input(i).set_data_type(mm.DataType.INT64).ok()
        network.get_input(i).set_dimension(input_dims).ok()
    return network

def get_args():
    # get common argparser,here is onnx_parser
    arg_parser = get_argparser()
    return arg_parser.parse_args()

def main():
    args = get_args()
    assert args.precision in ['force_float16', 'force_float32']
    network = get_network(args)
    config_args = extract_params("MODEL_CONFIG", args)
    config_network(network, config_args)
    builder_args = extract_params("MODEL_BUILDER", args)
    build_config = get_builder_config(builder_args)

    # generate model
    log.info('build model...')
    builder = mm.Builder()
    model = builder.build_model("bert_tensorflow_model", network, build_config)
    assert model is not None
    assert model.serialize_to_file(args.magicmind_model).ok()
    log.info("Generate model done, model save to %s" % args.magicmind_model) 
if __name__ == "__main__":
    main()

