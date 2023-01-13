import argparse
import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
from magicmind.python.runtime import DataType

def pytorch_parser(args):
    # 创建MagicMind parser
    parser = Parser(mm.ModelKind.kPytorch)
    # 设置网络输入数据类型
    parser.set_model_param("pytorch-input-dtypes", [DataType.FLOAT32])
    # 创建一个空的网络实例
    network = mm.Network()
    # 使用parser将Caffe模型文件转换为MagicMind Network实例。
    assert parser.parse(network, args.pt_model).ok()
    # 设置网络输入数据形状
    input_dims = mm.Dims((1, 1, 32, 200))
    assert network.get_input(0).set_dimension(input_dims).ok()
    return network

def generate_model_config(args):
    config = mm.BuilderConfig()
     # 指定硬件平台
    assert config.parse_from_string('{"archs":[{"mtp_372": [6,8]}]}').ok()
    # INT64 转 INT32
    assert config.parse_from_string('{"opt_config":{"type64to32_conversion": true}}').ok()
    assert config.parse_from_string('{"opt_config":{"conv_scale_fold": true}}').ok()
    assert config.parse_from_string('{"cross_compile_toolchain_path": "/tmp/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/"}').ok()
    # 模型输入输出规模可变功能
    assert config.parse_from_string('{"graph_shape_mutable": true}').ok()
    assert config.parse_from_string('{"dim_range": {"0": {"min": [1, 1, 32, 100], "max": [32, 1, 32, 300]}}}').ok()
    # 精度模式
    assert config.parse_from_string('{"precision_config":{"precision_mode":"%s"}}' % args.precision).ok()
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_model", "--pt_model", type=str, default="../data/models/crnn.pt", help="pt model")
    parser.add_argument("--output_model", "--output_model", type=str, default="../data/models/crnn_pytorch_model_force_float32_true_1", help="save mm model to this path")
    parser.add_argument("--precision", "--precision", type=str, default="qint8_mixed_float16", help="qint8_mixed_float16, force_float32, force_float16")
    args = parser.parse_args()
    
    network = pytorch_parser(args)
    config = generate_model_config(args)

    print('build model...')
    # 生成模型
    builder = mm.Builder()
    assert builder != None
    model = builder.build_model("crnn_model", network, config)
    assert model != None
    # 将模型序列化为离线文件
    assert model.serialize_to_file(args.output_model).ok()
    print("Generate model done, model save to %s" % args.output_model)

if __name__ == "__main__":
    main()
