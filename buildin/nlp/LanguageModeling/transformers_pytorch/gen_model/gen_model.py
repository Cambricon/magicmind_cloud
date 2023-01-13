import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
import numpy as np
import glob
import os 
import argparse
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "transformers model calibrartion and build")
    parser.add_argument('--precision', type=str,   default='force_float16', required=True ,help='Quant_mode')
    parser.add_argument('--batch_size', type=int,   default=8, required=True ,help='batch_size')
    parser.add_argument('--shape_mutable', type=str, default="", required=True ,help='shape_mutable')
    parser.add_argument('--pt_model', type=str, default="", required=True ,help='pt_model')
    parser.add_argument('--mm_model', type=str, default="", required=True ,help='mm_model')
    args = parser.parse_args()

    PYTORCH_MODEL_PATH = args.pt_model
    BATCH_SIZE = args.batch_size
    MAX_SEQ_LENGTH=128
    MM_MODEL=args.mm_model
    
    # step1: 创建MagicMind PyTorch parser
    mm_parser = Parser(mm.ModelKind.kPytorch)
    # step2: 设置网络输入数据类型
    mm_parser.set_model_param("pytorch-input-dtypes", [mm.DataType.INT32] * 3)
    # step3: 创建一个空的网络实例
    mm_network = mm.Network()
    # step4: 输入PyTorch模型文件，转换MagicMind网络
    status = mm_parser.parse(mm_network, PYTORCH_MODEL_PATH)
    assert status.ok()

    # 通过json字符串配置Builder参数
    config = mm.BuilderConfig()
    # INT64转INT32
    assert config.parse_from_string("{\"opt_config\":{\"type64to32_conversion\": true}}").ok()
    # 选用float16精度模式，支持float16及float32
    precision_json_str = '{"precision_config" : { "precision_mode" : "%s" }}'%args.precision
    assert config.parse_from_string(precision_json_str).ok()
    if args.shape_mutable=='true':
        assert config.parse_from_string('{"graph_shape_mutable": true}').ok()
        assert config.parse_from_string('{ \
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
        }}' % ((BATCH_SIZE, MAX_SEQ_LENGTH) * 3)).ok()
    else:
        assert config.parse_from_string('{"graph_shape_mutable": false}').ok()
    assert config.parse_from_string('{"archs":[{"mtp_372": [2,6,8]}]}').ok()


    # 设置模型输入形状和数据类型
    for i in range(mm_network.get_input_count()):
        mm_network.get_input(i).set_data_type(mm.DataType.INT32)
        mm_network.get_input(i).set_dimension(mm.Dims((BATCH_SIZE, MAX_SEQ_LENGTH)))

    # 生成模型
    mm_builder = mm.Builder()
    mm_model = mm_builder.build_model("bert", mm_network, config)
    assert mm_model is not None
    assert mm_model.serialize_to_file(MM_MODEL).ok()