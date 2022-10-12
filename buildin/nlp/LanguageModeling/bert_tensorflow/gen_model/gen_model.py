import json
import argparse
import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
def tensorflow_parser(args):
    # 创建MagicMind parser
    parser = Parser(mm.ModelKind.kTensorflow)
    parser.set_model_param("tf-infer-shape", False)
    parser.set_model_param("tf-model-type", "tf-graphdef-file")
    inputs_name = ['input_ids_1', 'input_mask_1', 'segment_ids_1']
    outputs_name = ['unstack:0', 'unstack:1']
    parser.set_model_param("tf-graphdef-inputs", inputs_name)
    parser.set_model_param("tf-graphdef-outputs", outputs_name)
    # 创建一个空的网络实例
    network = mm.Network()
    # 使用parser将TensorFlow模型文件转换为MagicMind Network实例。
    assert parser.parse(network, args.pb_model).ok()
    return network

def generate_model_config(args):
    config = mm.BuilderConfig()

    # 指定硬件平台
    assert config.parse_from_string('{"archs":[{"mtp_372": [8]}]}').ok()
    # INT64转INT32
    assert config.parse_from_string('{"opt_config":{"type64to32_conversion": true}}').ok()
    # 模型输入输出规模可变功能
    if args.shape_mutable == "true":
        dim_min = [1, args.max_seq_length]
        dim_max= [args.batch_size, args.max_seq_length]
        dim_range_value = {'0': {'min': dim_min, 'max': dim_max},'1': {'min': dim_min, 'max': dim_max},'2': {'min': dim_min, 'max': dim_max}}
        dim_str = json.dumps(dim_range_value, indent=2)
        print(dim_str)
        assert config.parse_from_string('{ \
        "graph_shape_mutable": true,  \
        "dim_range": %s \
        }' %(dim_str)).ok()
    else:
        assert config.parse_from_string('{"graph_shape_mutable":false}').ok()
    # 精度模式
    assert config.parse_from_string('{"precision_config":{"precision_mode":"%s"}}' % args.quant_mode).ok()
    return config

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--pb_model", "--pb_model", type=str, default="../data/models/frozen_graph.pb", help="bert pb")
    args.add_argument("--output_model", "--output_model", type=str, default="../data/models/bert_tensorflow_model", help="save mm model to this path")
    args.add_argument("--quant_mode", "--quant_mode", type=str, default="force_float16", help="force_float32, force_float16")
    args.add_argument("--shape_mutable", "--shape_mutable", type=str, default="true", help="whether the mm model is dynamic or static or not")
    args.add_argument("--batch_size", "--batch_size", type=int, default=16, help="batch_size")
    args.add_argument("--max_seq_length", "--max_seq_length", type=int, default=128, help="max_seq_length")
    args = args.parse_args()

    supported_quant_mode = ['force_float16', 'force_float32']
    if args.quant_mode not in supported_quant_mode:
        print('quant_mode [' + args.quant_mode + ']', 'not supported')
        exit()

    network = tensorflow_parser(args)
    config = generate_model_config(args)
    # 生成模型
    print('build model...')
    builder = mm.Builder()
    model = builder.build_model("bert_tensorflow_model", network, config)
    assert model is not None
    # 将模型序列化为离线文件
    assert model.serialize_to_file(args.output_model).ok()
    print("Generate model done, model save to %s" % args.output_model) 
if __name__ == "__main__":
    main()
