from magicmind.python.runtime.parser import Parser
import magicmind.python.runtime as mm

import os
import numpy as np
from typing import List
import argparse
from calibrator import CalibData

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "c3d caffe model calibrartion and build")
    parser.add_argument('--precision', type=str,   default='', required=True ,help='Quant_mode')
    parser.add_argument('--batch_size', type=int,   default=8, required=True ,help='batch_size')
    parser.add_argument('--shape_mutable', type=str, default="", required=True ,help='shape_mutable')
    parser.add_argument('--datasets_dir', type=str, default="", required=True ,help='datasets_dir')
    parser.add_argument('--caffe_prototxt', type=str, default="", required=True ,help='caffe_prototxt')
    parser.add_argument('--caffe_model', type=str, default="", required=True ,help='caffe_model')
    parser.add_argument('--mm_model', type=str, default="", required=True ,help='mm_model')
    args = parser.parse_args()

    DATASET_DIR = args.datasets_dir
    PROTOTXT = args.caffe_prototxt 
    CAFFEMODEL = args.caffe_model
    DEV_ID = 0
    # 校验MD5
    md5 = os.popen('md5sum ' + CAFFEMODEL).readline().split()[0]
    assert md5 == '273e01fb6b7f48b6c0bdb5360fb7fdac'
    # 模型输入规模
    BATCH_SIZE = args.batch_size
    INPUT_SIZE = (112, 112) # h x w
    CLIP_LEN = 8 # 视频片段长度
    # 网络预处理将图片resize到 RESIZE_WIDTH * RESIZE_HEIGHT再做CenterCrop到INPUT_SIZE
    RESIZE_WIDTH = 171
    RESIZE_HEIGHT = 128
    # MLU设备id

    #${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
    # 设置生成的magicmind模型路径
    MM_MODEL = args.mm_model
    # 创建MagicMind parser
    parser = Parser(mm.ModelKind.kCaffe)
    # 生成Network实例
    network = mm.Network()
    assert parser.parse(network, CAFFEMODEL, PROTOTXT).ok()
    # 设置网络输入数据形状
    network.get_input(0).set_dimension(mm.Dims((BATCH_SIZE, 3, CLIP_LEN, INPUT_SIZE[0], INPUT_SIZE[1])))

    config = mm.BuilderConfig()
    precision_json_str = '{"precision_config" : { "precision_mode" : "%s" }}'%args.precision
    assert config.parse_from_string(precision_json_str).ok()
    assert config.parse_from_string("{\"opt_config\":{\"type64to32_conversion\":true}}").ok()
    assert config.parse_from_string("{\"opt_config\":{\"conv_scale_fold\":true}}").ok()
    # 禁用模型输入输出规模可变功能
    if args.shape_mutable=='true':
        assert config.parse_from_string('{"graph_shape_mutable": true}').ok()
        assert config.parse_from_string('{"dim_range": {"0": {"min": [1,3,8,112,112], "max": [32,3,8,112,112]}}}').ok()
    else:
        assert config.parse_from_string('{"graph_shape_mutable": false}').ok()

    assert config.parse_from_string("""{"cross_compile_toolchain_path": "/tmp/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/"}""").ok()
    # 硬件平台
    assert config.parse_from_string("""{"archs": ["mtp_372"]}""").ok()
    assert config.parse_from_string('{"convert_input_layout": { "0": {"src": "NCDHW", "dst": "NDHWC"}}}').ok()
    # 量化算法，支持对称量化（symmetric)和非对称量化（asymmetric）。当量化统计算法设置为EQNM_ALOGORITHM时，仅适用于对称量化。
    assert config.parse_from_string('{"precision_config": {"activation_quant_algo": "symmetric"}}').ok()
    # 设置量化粒度，支持按tensor量化（per_tensor）和按通道量化（per_axis）两种。
    assert config.parse_from_string('{"precision_config": {"weight_quant_granularity": "per_tensor"}}').ok()

    if args.precision != "force_float16" and args.precision != "force_float32":
        # 样本数据所在目录(目录中需存放若干jpg格式图片)
        CALIB_VIDEO_LIST = [os.path.join(DATASET_DIR,'WritingOnBoard/v_WritingOnBoard_g21_c06.avi'),
                            os.path.join(DATASET_DIR,'WalkingWithDog/v_WalkingWithDog_g16_c04.avi'),
                            os.path.join(DATASET_DIR,'WritingOnBoard/v_WritingOnBoard_g19_c01.avi'),
                            os.path.join(DATASET_DIR,'WallPushups/v_WallPushups_g20_c04.avi')]
        # 最大样本数量
        MAX_CALIB_SAMPLES = BATCH_SIZE
        # 创建量化工具并设置量化统计算法
        calib_data = CalibData(mm.Dims((BATCH_SIZE, 3, CLIP_LEN, INPUT_SIZE[0], INPUT_SIZE[1])), MAX_CALIB_SAMPLES, RESIZE_WIDTH,RESIZE_HEIGHT,CALIB_VIDEO_LIST)
        calibrator = mm.Calibrator([calib_data])
        assert calibrator is not None
        # 设置量化统计算法，支持线性统计算法（LINEAR_ALGORITHM）及加强的最小化量化噪声算法（EQM_ALGORITHM）。
        assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()

    # 打开设备
    with mm.System() as mm_sys:
        dev_count = mm_sys.device_count()
        print("Device count: ", dev_count)
        if DEV_ID >= dev_count:
            print("Invalid DEV_ID set!")
        # 打开MLU设备
        dev = mm.Device()
        dev.id = DEV_ID
        assert dev.active().ok()

    # 进行量化
    if args.precision != "force_float16" and args.precision != "force_float32":
        print("Calibrating......")
        assert calibrator.calibrate(network, config).ok()
        print("Calibrating done!")

    # 生成模型
    builder = mm.Builder()
    assert builder is not None
    mm_model = builder.build_model("magicmind model", network, config)
    assert mm_model is not None
    # 将模型序列化为离线文件
    assert mm_model.serialize_to_file(MM_MODEL).ok()
