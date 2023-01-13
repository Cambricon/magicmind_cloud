from calendar import c
import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
import argparse
from calibrator import CalibData

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "model calibrartion and build")
    parser.add_argument('--precision', type=str,   default='force_float16', required=True ,help='Quant_mode')
    parser.add_argument('--batch_size', type=int,   default=8, required=True ,help='batch_size')
    parser.add_argument('--shape_mutable', type=str, default="", required=True ,help='shape_mutable')
    parser.add_argument('--datasets_dir', type=str, default="", required=True ,help='datasets_dir')
    parser.add_argument('--pt_model', type=str, default="", required=True ,help='pt_model')
    parser.add_argument('--mm_model', type=str, default="", required=True ,help='magicmind model')
    args = parser.parse_args()

    PT_MODEL = args.pt_model
    BATCH_SIZE = args.batch_size
    MM_MODEL = args.mm_model
    DEV_ID = 0
    MEAN = [123.675, 116.28 , 103.53]
    STD = [58.395, 57.12 , 57.375]
    INPUT_SIZE = (224, 224)
    NEW_SIZE = 256
    assert NEW_SIZE >= INPUT_SIZE[0]
    assert NEW_SIZE >= INPUT_SIZE[1]

    # 创建MagicMind parser
    parser = Parser(mm.ModelKind.kPytorch)
    # 指定网络输入数据类型
    parser.set_model_param('pytorch-input-dtypes', [mm.DataType.FLOAT32])
    network = mm.Network()
    assert parser.parse(network,PT_MODEL).ok()
    assert network.get_input(0).set_dimension(mm.Dims((BATCH_SIZE, 3, INPUT_SIZE[0], INPUT_SIZE[1]))).ok()
    config = mm.BuilderConfig()
    precision_json_str = '{"precision_config" : { "precision_mode" : "%s" }}'%args.precision
    assert config.parse_from_string(precision_json_str).ok()
    assert config.parse_from_string("{\"opt_config\":{\"type64to32_conversion\":true}}").ok()
    assert config.parse_from_string("{\"opt_config\":{\"conv_scale_fold\":true}}").ok()
    # 禁用模型输入输出规模可变功能
    if args.shape_mutable=='true':
        assert config.parse_from_string('{"graph_shape_mutable": true}').ok()
        assert config.parse_from_string('{"dim_range": {"0": {"min": [1, 3, 224, 224], "max": [64, 3, 224, 224]}}}').ok()
    else:
        assert config.parse_from_string('{"graph_shape_mutable": false}').ok()
    assert config.parse_from_string('{"archs":[{"mtp_372": [2,6,8]}]}').ok()
    assert config.parse_from_string('{"convert_input_layout": { "0": {"src": "NCHW", "dst": "NHWC"}}}').ok()
    if args.precision != "force_float16" and args.precision != "force_float32":
        # 量化算法，支持对称量化（symmetric)和非对称量化（asymmetric）。当量化统计算法设置为EQNM_ALOGORITHM时，仅适用于对称量化。
        assert config.parse_from_string('{"precision_config": {"activation_quant_algo": "symmetric"}}').ok()
        # 设置量化粒度，支持按tensor量化（per_tensor）和按通道量化（per_axis）两种。
        assert config.parse_from_string('{"precision_config": {"weight_quant_granularity": "per_tensor"}}').ok()

    need_insert_bn = False
    for v in MEAN:
        if v != 0:
            need_insert_bn = True
            break
    if not need_insert_bn:
        for v in STD:
            if v != 1.0:
                need_insert_bn = True
                break
    if need_insert_bn:
        var = list(map(lambda num:num*num, STD))
        assert config.parse_from_string('{"insert_bn_before_firstnode": {"0": {"mean":' + str(MEAN) + \
                                        ', "var": ' + str(var) + '}}}').ok()
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
        if args.precision != "force_float16" and args.precision != "force_float32":
            print("Calibrating...")
            # 样本数据所在目录(目录中需存放若干jpg格式图片)
            CALIB_SAMPLES_DIR = args.datasets_dir
            # 最大样本数量
            MAX_CALIB_SAMPLES = BATCH_SIZE
            # 创建量化工具并设置量化统计算法
            calib_data = CalibData(mm.Dims((BATCH_SIZE, 3, INPUT_SIZE[0], INPUT_SIZE[1])), MAX_CALIB_SAMPLES, CALIB_SAMPLES_DIR,\
                                            new_size=NEW_SIZE,need_insert_bn_= need_insert_bn,MEAN_=MEAN,STD_=STD)
            calibrator = mm.Calibrator([calib_data])
            assert calibrator is not None
            # 设置量化统计算法，支持线性统计算法（LINEAR_ALGORITHM）及加强的最小化量化噪声算法（EQM_ALGORITHM）。
            assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()
            # 进行量化
            assert calibrator.calibrate(network, config).ok()
            print("Calibrate Done!")
        if not need_insert_bn:
            # 当使用了insert_bn_before_firstnode参数后，不需要设置网络输入数据类型，且生成的网络的输入数据类型为UINT8。
            # 本例程序使用UINT8数据类型作为输入，故此处设置网络输入类型为UINT8。
            network.get_input(0).set_data_type(mm.DataType.UINT8)
        # 生成模型
        builder = mm.Builder()
        assert builder is not None
        mm_model = builder.build_model("model", network, config)
        assert mm_model is not None
        # 将模型序列化为离线文件
        assert mm_model.serialize_to_file(MM_MODEL).ok()