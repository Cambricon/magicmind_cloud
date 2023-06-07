import json
import argparse
import magicmind.python.runtime as mm
from framework_parser import TensorFlowParser
from build_param import get_argparser 
from model_process import (
    extract_params,
    config_network,
    get_builder_config,
    build_and_serialize,
)
from logger import Logger
log = Logger()
import tensorflow as tf
log.info(tf.__version__)
log.info("tf version is:tf{}".format(tf.__version__[0]))


def get_network(args):
    parser = TensorFlowParser(args)
    network = parser.parse()
    input_dims = mm.Dims((args.input_dims[0]))
    assert network.get_input(0).set_dimension(input_dims).ok()
    assert network.get_input(1).set_dimension(input_dims).ok()
    return network

def get_args():
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
