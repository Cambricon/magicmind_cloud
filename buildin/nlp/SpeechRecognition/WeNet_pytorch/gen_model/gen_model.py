import os
import json
import argparse
import numpy as np
import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser

def onnx_parser(onnx_model):
    # 创建MagicMind parser
    parser = Parser(mm.ModelKind.kOnnx)
    # 使用parser将 Onnx 模型文件转换为MagicMind Network实例。
    network = mm.Network()
    assert parser.parse(network, onnx_model).ok()
    return network

def get_model_config(config_file):
    config = mm.BuilderConfig()
    with open(config_file) as f:
        config_str = f.read()
        print(config_str)
    assert config.parse_from_string(config_str).ok()
    return config

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--encoder", type=str, default="../data/models/20211025_conformer_exp/onnx_model/encoder.onnx", help="encoder onnx file")
    args.add_argument("--decoder", type=str, default="../data/models/20211025_conformer_exp/onnx_model/decoder.onnx", help="decoder onnx file")
    args.add_argument("--json", type=str, default="../data/json/builder_config.json", help="builder config file")
    args.add_argument("--precision", type=str, default="force_float32", help="only support force_float32")
    args.add_argument("--output", type=str,  default="../data/models/", required=True, help="output folder to save mm model")
    args = args.parse_args()

    output_model_encoder = args.output+"/"+"encoder_"+args.precision+"_model"
    output_model_decoder = args.output+"/"+"decoder_"+args.precision+"_model"

    network = onnx_parser(args.encoder)
    config  = get_model_config(args.json)
    supported_precision = ['force_float32']
    if args.precision not in supported_precision:
        print('precision [' + args.precision + ']', 'not supported')
        exit()

     # 生成模型
    print('build encoder model...')
    builder = mm.Builder()
    model = builder.build_model('encoder', network, config)
    assert model is not None
    # 将模型序列化为离线文件
    assert model.serialize_to_file(output_model_encoder).ok()
    print("Generate model done, model save to %s" % output_model_encoder)
    
    network = onnx_parser(args.decoder)
     # 生成模型
    print('build decoder model...')
    builder = mm.Builder()
    model = builder.build_model('decoder', network, config)
    assert model is not None
    # 将模型序列化为离线文件
    assert model.serialize_to_file(output_model_decoder).ok()
    print("Generate model done, model save to %s" % output_model_decoder)

if __name__ == "__main__":
    main()
