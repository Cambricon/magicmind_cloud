import argparse
import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
def pytorch_parser(args):
    # 创建MagicMind parser
    parser = Parser(mm.ModelKind.kPytorch)
    # 设置网络输入数据类型
    parser.set_model_param("pytorch-input-dtypes", [mm.DataType.INT32] * 3)
    # 创建一个空的网络实例
    network = mm.Network()
    # 使用parser将PyTorch模型文件转换为MagicMind Network实例。
    assert parser.parse(network, args.pt_model).ok()
    # 设置模型输入形状和数据类型
    for i in range(network.get_input_count()):
        network.get_input(i).set_data_type(mm.DataType.INT32)
        network.get_input(i).set_dimension(mm.Dims((args.batch_size, args.max_seq_length)))
    return network

def generate_model_config(args):
    config = mm.BuilderConfig()

    # 指定硬件平台
    assert config.parse_from_string('{"archs":[{"mtp_372": [6,8]}]}').ok()
    # INT64转INT32
    assert config.parse_from_string('{"opt_config":{"type64to32_conversion": true}}').ok()
    # 模型输入输出规模可变功能
    if args.shape_mutable == "true":
        assert config.parse_from_string('{ \
        "graph_shape_mutable": true,  \
        "dim_range": {  \
        "0": {  \
            "min": [1, 1],  \
            "max": [%d, %d]  \
        },  \
        "1": {  \
            "min": [1, 1],  \
            "max": [%d, %d]  \
        },  \
        "2": {  \
            "min": [1, 1],  \
            "max": [%d, %d]  \
        }  \
        }}' % ((64, args.max_seq_length) * 3)).ok()
    else:
        assert config.parse_from_string('{"graph_shape_mutable":false}').ok()
    # 精度模式
    assert config.parse_from_string('{"precision_config":{"precision_mode":"%s"}}' % args.quant_mode).ok()
    assert config.parse_from_string('{"compute_determinism": true}').ok()
    return config

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--pt_model", "--pt_model", type=str, default="../data/models/bert_qa_pytorch.pt", help="bert_qa_pytorch pt")
    args.add_argument("--output_model", "--output_model", type=str, default="../data/models/bert_qa_pytorch_model", help="save mm model to this path")
    args.add_argument("--quant_mode", "--quant_mode", type=str, default="force_float16", help="force_float32, force_float16")
    args.add_argument("--shape_mutable", "--shape_mutable", type=str, default="true", help="whether the mm model is dynamic or static or not")
    args.add_argument("--batch_size", "--batch_size", type=int, default=16, help="batch_size")
    args.add_argument("--max_seq_length", "--max_seq_length", type=int, default=128, help="max_seq_length")
    args = args.parse_args()

    supported_quant_mode = ['force_float16', 'force_float32']
    if args.quant_mode not in supported_quant_mode:
        print('quant_mode [' + args.quant_mode + ']', 'not supported')
        exit()

    network = pytorch_parser(args)
    config = generate_model_config(args)
    # 生成模型
    print('build model...')
    builder = mm.Builder()
    model = builder.build_model("bert_pytorch_model", network, config)
    assert model is not None
    # 将模型序列化为离线文件
    assert model.serialize_to_file(args.output_model).ok()
    print("Generate model done, model save to %s" % args.output_model) 
if __name__ == "__main__":
    main()
