import argparse
import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
from magicmind.python.runtime import DataType
from calibrator import CalibData

def pytorch_parser(args):
    # 创建MagicMind parser
    parser = Parser(mm.ModelKind.kPytorch)
    # 设置网络输入数据类型
    parser.set_model_param("pytorch-input-dtypes", [DataType.FLOAT32])
    # 创建一个空的网络实例
    network = mm.Network()
    # 使用parser将PyTorch模型文件转换为MagicMind Network实例。
    assert parser.parse(network, args.pt_model).ok()
    # 设置网络输入数据形状
    input_dims = mm.Dims((args.batch_size, 3, 16, 112, 112))
    assert network.get_input(0).set_dimension(input_dims).ok()
    return network

def generate_model_config(args):
    config = mm.BuilderConfig()
     # 指定硬件平台
    assert config.parse_from_string('{"archs":[{"mtp_372": [2,6,8]}]}').ok()
    # INT64 转 INT32
    assert config.parse_from_string('{"opt_config":{"type64to32_conversion": true}}').ok()
    assert config.parse_from_string('{"opt_config":{"conv_scale_fold": true}}').ok()
    # 模型输入输出规模可变功能
    if args.shape_mutable == "true":
        assert config.parse_from_string('{"graph_shape_mutable": true}').ok()
        assert config.parse_from_string('{"dim_range": {"0": {"min": [1, 3, 16, 112, 112], "max": [64, 3, 16, 112, 112]}}}').ok()
    else:
        assert config.parse_from_string('{"graph_shape_mutable": false}').ok()
    # 精度模式
    assert config.parse_from_string('{"precision_config":{"precision_mode":"%s"}}' % args.precision).ok()
    # 量化算法，支持对称量化（symmetric)和非对称量化（asymmetric）。当量化统计算法设置为EQNM_ALOGORITHM时，仅适用于对称量化。
    assert config.parse_from_string('{"precision_config": {"activation_quant_algo": "symmetric"}}').ok()
    # 设置量化粒度，支持按tensor量化（per_tensor）和按通道量化（per_axis）两种。
    assert config.parse_from_string('{"precision_config": {"weight_quant_granularity": "per_tensor"}}').ok()
    return config

def calibrate(args, network : mm.Network, config : mm.BuilderConfig):
    # 创建量化工具并设置量化统计算法
    calib_data = CalibData(shape = mm.Dims([args.batch_size, 3, 6, 112, 112]), max_samples = args.batch_size, img_dir = args.image_dir)
    calibrator = mm.Calibrator([calib_data])
    assert calibrator is not None
    # 设置量化统计算法，支持线性统计算法（LINEAR_ALGORITHM）及加强的最小化量化噪声算法（EQM_ALGORITHM）。
    assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()
    # 打开设备
    with mm.System() as mm_sys:
        dev_count = mm_sys.device_count()
        print("Device count: ", dev_count)
        if args.device_id >= dev_count:
            print("Invalid device set!")
            # 打开MLU设备
            dev = mm.Device()
            dev.id = args.device_id
            assert dev.active().ok()
    # 进行量化
    assert calibrator.calibrate(network, config).ok()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", dest = 'device_id', default = 0, type = int, help = 'mlu device id, used for calibration')
    parser.add_argument("--pt_model", "--pt_model", type=str, default="../data/models/3dresnet_traced.pt", help="modified yolov5s pt")
    parser.add_argument("--output_model", "--output_model", type=str, default="../data/models/3dresnet_pytorch_model", help="save mm model to this path")
    parser.add_argument("--image_dir", "--image_dir",  type=str, default="../data/kinetics_videos/jpg/", help="kinetics datasets")
    parser.add_argument("--precision", "--precision", type=str, default="qint8_mixed_float16", help="qint8_mixed_float16, force_float32, force_float16")
    parser.add_argument("--shape_mutable", "--shape_mutable", type=str, default="false", help="whether the mm model is dynamic or static or not")
    parser.add_argument("--batch_size", "--batch_size", type=int, default=1, help="batch_size")
    args = parser.parse_args()

    network = pytorch_parser(args)
    config = generate_model_config(args)

    if args.precision.find('qint') != -1:
        print('do calibrate...')
        calibrate(args, network, config)
    print('build model...')
    # 生成模型
    builder = mm.Builder()
    model = builder.build_model("3dresnet_pytorch_model", network, config)
    assert model != None
    # 将模型序列化为离线文件
    model.serialize_to_file(args.output_model)
    print("Generate model done, model save to %s" % args.output_model)

if __name__ == "__main__":
    main()
