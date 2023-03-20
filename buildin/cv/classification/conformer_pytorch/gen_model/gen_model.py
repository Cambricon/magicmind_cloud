import argparse
import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
from magicmind.python.runtime import DataType
from calibrator import CalibData
from typing import List

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
    input_dims = mm.Dims((args.batch_size, 3, args.input_height, args.input_width))
    assert network.get_input(0).set_dimension(input_dims).ok()
    return network

def generate_model_config(args):
    config = mm.BuilderConfig()
    # 指定硬件平台
    # assert config.parse_from_string('{"archs":[{"mtp_372": [2,6,8]}]}').ok()
    assert config.parse_from_string('{"archs":[{"mtp_372": [6]}]}').ok()
    # INT64 转 INT32
    assert config.parse_from_string('{"opt_config":{"type64to32_conversion": true}}').ok()
    assert config.parse_from_string('{"opt_config":{"conv_scale_fold": true}}').ok()
    # 输入数据摆放顺序
    # 模型输入数据顺序为NCHW，如下代码转为NHWC输入顺序。
    # 输入顺序的改变需要同步到推理过程中的网络预处理实现，保证预处理结果的输入顺序与网络输入数据顺序一致。
    # 以下JSON字符串中的0代表改变的是网络第一个输入的数据摆放顺序。1则代表第二个输入，以此类推。
    assert config.parse_from_string('{"convert_input_layout": {"0": {"src": "NCHW", "dst": "NHWC"}}}').ok()
    # 模型输入输出规模可变功能
    if args.shape_mutable == "true":
        assert config.parse_from_string('{"graph_shape_mutable": true}').ok()
        assert config.parse_from_string('{"dim_range": {"0": {"min": [1, 3, 224, 224], "max": [512, 3, 224, 224]}}}').ok()
    else:
        assert config.parse_from_string('{"graph_shape_mutable": false}').ok()
    # 高精度模式
    assert config.parse_from_string('{"computation_preference": "fast"}').ok()
    # 精度模式
    assert config.parse_from_string('{"precision_config":{"precision_mode":"%s"}}' % args.precision).ok()
    # 预处理标准化过程（减均值、除标准差: (input - mean) / std）集成到模型中, 推理过程的预处理代码中就不再需要执行标准化相关计算。
    # 以下JSON字符串中键值0代表处理的是网络的第一个输入。mean数组中的值代表输入各通道的均值。var数组中的值代表输入各通道的方差即std²。
    # 预处理标准化过程放入模型中后，在模型生成章节中可把模型输入数据类型设置为UINT8，从而达到减少输入拷贝量的目的。
    # 该功能只适用于首层为卷积的网络。
    if args.need_insert_bn:
        # MEAN = [116.28, 103.53, 123.675]
        # STD = [57.12, 57.375, 58.395]
        MEAN = [123.675, 116.28, 103.53]
        STD = [58.395, 57.12, 57.375]
        var = list(map(lambda num:num*num, STD))
        assert config.parse_from_string('{"insert_bn_before_firstnode": {"0": {"mean":' + str(MEAN) + \
                                        ', "var": ' + str(var) + '}}}').ok()
    # 量化算法，支持对称量化（symmetric)和非对称量化（asymmetric）。当量化统计算法设置为EQNM_ALOGORITHM时，仅适用于对称量化。
    assert config.parse_from_string('{"precision_config": {"activation_quant_algo": "symmetric"}}').ok()
    # 设置量化粒度，支持按tensor量化（per_tensor）和按通道量化（per_axis）两种。
    assert config.parse_from_string('{"precision_config": {"weight_quant_granularity": "per_tensor"}}').ok()
    return config, MEAN, STD

def calibrate(args, network: mm.Network, config: mm.BuilderConfig, mean:List[float], std:List[float]):
    calib_data = CalibData(mm.Dims((args.batch_size, 3, args.input_height, args.input_width)), args.batch_size, args.image_dir, args.need_insert_bn, mean, std)
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
            abort()
        # 打开MLU设备
        dev = mm.Device()
        dev.id = args.device_id
        assert dev.active().ok()
        print("Wroking on MLU ", args.device_id)
    # 进行量化
    assert calibrator.calibrate(network, config).ok()

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--pt_model", "--pt_model", type=str, default="xxx.pt", help="original pt")
    args.add_argument("--output_model", "--output_model", type=str, default="magicmind_model", help="save mm model to this path")
    args.add_argument("--image_dir", "--image_dir",  type=str, default="ILSVRC2012", help="datasets")
    args.add_argument("--label_file", "--label_file", type=str, default="ILSVRC2012_val.txt")
    args.add_argument("--precision", "--precision", type=str, default="force_float32", help="force_float32, force_float16")
    args.add_argument("--shape_mutable", "--shape_mutable", type=str, default="false", help="whether the mm model is dynamic or static or not")
    args.add_argument('--batch_size', dest = 'batch_size', default = 1, type = int, help = 'batch_size')
    args.add_argument('--input_width', dest = 'input_width', default = 224, type = int, help = 'model input width')
    args.add_argument('--input_height', dest = 'input_height', default = 224, type = int, help = 'model input height')
    args.add_argument('--device_id', dest = 'device_id', default = 0, type = int, help = 'device_id')
    args.add_argument('--need_insert_bn', default=True, type=bool)
    args = args.parse_args()

    network = pytorch_parser(args)
    config, mean, std = generate_model_config(args)

    if args.precision.find('qint') != -1:
        print('do calibrate...')
        calibrate(args, network, config, mean, std)

    if not args.need_insert_bn:
        # 当使用了insert_bn_before_firstnode参数后，不需要设置网络输入数据类型，且生成的网络的输入数据类型为UINT8。
        # 本例程序使用UINT8数据类型作为输入，故此处设置网络输入类型为UINT8。
        network.get_input(0).set_data_type(mm.DataType.UINT8)

    # 生成模型
    print('build model...')
    builder = mm.Builder()
    assert builder != None
    model = builder.build_model("conformer_small_patch16_model", network, config)
    assert model != None
    # 将模型序列化为离线文件
    assert model.serialize_to_file(args.output_model).ok()
    # 检查模型输入数据类型是否为UINT8
    print("input_dtype of this model:", model.get_input_data_type(0))
    print("Generate model done, model save to %s" % args.output_model)

if __name__ == "__main__":
    main()

