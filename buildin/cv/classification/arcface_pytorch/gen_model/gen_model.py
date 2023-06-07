import magicmind.python.runtime as mm
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

log = Logger()

def get_network(args):
    parser = PytorchParser(args)
    network = parser.parse()
    return network

def calibrate(args, network : mm.Network, config : mm.BuilderConfig): 
    input_dims = mm.Dims(args.input_dims[0])
    calib_data = CalibData(shape=input_dims, img_dir=args.image_dir)
    common_calibrate(args, network, config, calib_data)

def get_args():
    # get common argparser,here is pytorch_parser
    arg_parser = get_argparser()
    # add custom args belonging to the current net.
    arg_parser.add_argument(
        "--image_dir",
        dest="image_dir",
        type=str,
        default="./file_list.txt",
        help="calib file list",
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
    build_and_serialize_args = extract_params("MODEL_BUILD_AND_SERIALIZE", args)
    build_and_serialize(network, build_config, build_and_serialize_args)

    end_time = time.time()
    log.info("gen_model time cost:{:.3f}s".format(end_time - begin_time))    

if __name__ == "__main__":
    main()

