import magicmind.python.runtime as mm
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
# 实例化python logger类
log = Logger()

def calibrate(args, network : mm.Network, config : mm.BuilderConfig):
    custom_max_samples = 10
    max_samples = max(custom_max_samples, args.input_dims[0][0])
    calib_data = CalibData(args, mm.Dims([args.input_dims[0][0], args.input_dims[0][1], args.input_dims[0][2], args.input_dims[0][3]]), 
                            max_samples, args.image_dir)
    common_calibrate(args, network, config, calib_data)

def get_network(args):
    parser = OnnxParser(args)
    network = parser.parse()

    return network

def get_args():
    # get common argparser,here is pytorch_parser
    arg_parser = get_argparser()

    # add custom args belonging to the current net.
    arg_parser.add_argument("--task", type=str, default="det", help="det, rec or cls")
    
    # params for text detection    
    arg_parser.add_argument("--image_dir", type=str, default="../doc/imgs/11.jpg", help='det_image_dir')
    arg_parser.add_argument("--det_batch_num", type=int, default=1, help='det_batch_num')
    arg_parser.add_argument("--det_limit_side_len", type=float, default=1280, help='det_limit_side_len')
    arg_parser.add_argument("--det_limit_type", type=str, default='max', help='det_limit_type')

    # params for text recognizer    
    # arg_parser.add_argument("--rec_image_shape", type=str, default="3, 48, 320", help='rec_image_shape')
    arg_parser.add_argument("--rec_batch_num", type=int, default=6, help='rec_batch_num,default is 6')
    arg_parser.add_argument("--limited_max_width", type=int, default=1280, help='limited_max_width')
    arg_parser.add_argument("--limited_min_width", type=int, default=16, help='limited_min_width')

    #params for text classifiler
    arg_parser.add_argument("--cls_model_path", type=str,default="", help='cls_model_path,default is:../data/models/ch_ppocr_mobile_v2.0_cls_infer.onnx')
    arg_parser.add_argument("--cls_batch_num", type=int,default=6,  help='cls_batch_num,default is 6')
    # arg_parser.add_argument("--cls_image_shape",type=str, default="3, 48, 192")
    args = arg_parser.parse_args()

    if args.task not in ['det', 'rec', 'cls']:
        log.err("Please input  det, rec or cls!")


    if args.precision == 'qint8_mixed_float16' and args.task == 'rec':
        log.err("OCR REC model not support qint8_mixed_float16!!! set 'force_float32' or 'force_float16' !")

    return args

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

    if args.precision.find('qint') != -1:
        log.info('Do calibrate...')
        calibrate(args, network, build_config)
    log.info('build model...')
    # 生成模型并导出
    build_and_serialize_args = extract_params("MODEL_BUILD_AND_SERIALIZE", args)
    build_and_serialize(network, build_config, build_and_serialize_args)

    end_time = time.time()
    log.info("gen_model time cost:{:.3f}s".format(end_time - begin_time))

if __name__ == '__main__':
    main()
