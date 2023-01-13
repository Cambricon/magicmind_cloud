import argparse

from magicmind.python.runtime.parser import Parser
from magicmind.python.runtime import Network, Builder, Model
from magicmind.python.runtime import BuilderConfig, ModelKind, Dims

import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from mm_utils import _check_status, parse_dynamic_size

def build_mm(model_file, model_name, dim_range_string, shape_mutable, precision):
    network = Network()
    builder = Builder()
    config = BuilderConfig()
    
    # 创建MagicMind parser
    parser = Parser(ModelKind.kOnnx)
    # 使用parser将ONNX模型文件转换为MagicMind Network实例。
    assert parser.parse(network, model_file).ok()

    # 指定硬件平台
    assert config.parse_from_string('{"archs":[{"mtp_372": [6,8]}]}').ok()
    # 精度模式
    assert config.parse_from_string('{"precision_config":{"precision_mode":"%s"}}' % precision).ok()
    # INT64 转 INT32
    assert config.parse_from_string('{"opt_config":{"type64to32_conversion":true}}').ok()
    assert config.parse_from_string('{"opt_config":{"conv_scale_fold":true}}').ok()
    if shape_mutable == "true":
        assert config.parse_from_string('{"graph_shape_mutable":true}').ok()
        if dim_range_string:
            config.parse_from_string(dim_range_string)
    else:
        assert config.parse_from_string('{"graph_shape_mutable":false}').ok()
    model = builder.build_model(model_name, network, config)
    return model

def parse_args(parser):
    """
      Parse commandline arguments.
    """
    parser.add_argument('-o', '--output', required=True,
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('--encoder', type=str, default="",
                        help='full path to the Encoder ONNX')
    parser.add_argument('--decoder', type=str, default="",
                        help='full path to the DecoderIter ONNX')
    parser.add_argument('--postnet', type=str, default="",
                        help='full path to the Postnet ONNX')
    parser.add_argument('--waveglow', type=str, default="",
                        help='full path to the WaveGlow ONNX')
    parser.add_argument('--precision', type=str, default="force_float32",
                        help='force_float32, force_float16')
    parser.add_argument('--shape_mutable', type=str, default="true",
                        help='true, false')
    parser.add_argument('-bs', '--batch-size', type=str, default="1",
                        help='One or three comma separated integers specifying the batch size. Specify "min,opt,max" for dynamic shape')
    parser.add_argument('--mel-size', type=str, default="32,768,1664",
                        help='One or three comma separated integers specifying the mels size for waveglow.')
    parser.add_argument('--z-size', type=str, default="1024,24576,53248",
                        help='One or three comma separated integers specifying the z size for waveglow.')
  
    return parser


def main():
    parser = argparse.ArgumentParser(
          description='Export from ONNX to MM for Tacotron 2 and WaveGlow')
    parser = parse_args(parser)
    args = parser.parse_args()

    path=args.output.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)

    bs_min, bs_opt, bs_max = parse_dynamic_size(args.batch_size)
    mel_min, mel_opt, mel_max = parse_dynamic_size(args.mel_size)
    z_min, z_opt, z_max = parse_dynamic_size(args.z_size)

    encoder_dim_range = None
    decoder_dim_range = None
    postnet_dim_range = None
    waveglow_dim_range = None

    encoder_dim_range = "{\"dim_range\": {\"0\": {\"min\": [" + str(bs_min) + ", 4], \"max\": [" + str(bs_max) + ", 256]}, \
             \"1\": {\"min\": [" + str(bs_min) + " ], \"max\": [" + str(bs_max) + "]}}}"
    decoder_dim_range = "{\"dim_range\": {\"0\": {\"min\": [" + str(bs_min) + ", 80], \"max\": [" + str(bs_max) + ", 80]}, \
      \"1\": {\"min\": [" + str(bs_min) + ", 1024], \"max\": [" + str(bs_max) + ", 1024]}, \
      \"2\": {\"min\": [" + str(bs_min) + ", 1024], \"max\": [" + str(bs_max) + ", 1024]}, \
      \"3\": {\"min\": [" + str(bs_min) + ", 1024], \"max\": [" + str(bs_max) + ", 1024]}, \
      \"4\": {\"min\": [" + str(bs_min) + ", 1024], \"max\": [" + str(bs_max) + ", 1024]}, \
      \"5\": {\"min\": [" + str(bs_min) + ", 4], \"max\": [" + str(bs_max) + ", 256]}, \
      \"6\": {\"min\": [" + str(bs_min) + ", 4], \"max\": [" + str(bs_max) + ", 256]}, \
      \"7\": {\"min\": [" + str(bs_min) + ", 512], \"max\": [" + str(bs_max) + ", 512]}, \
      \"8\": {\"min\": [" + str(bs_min) + ", 4, 512], \"max\": [" + str(bs_max) + ", 256, 512]}, \
      \"9\": {\"min\": [" + str(bs_min) + ", 4, 128], \"max\": [" + str(bs_max) + ", 256, 128]}, \
      \"10\": {\"min\": [" + str(bs_min) + ", 4], \"max\": [" + str(bs_max) + ", 256]}}}"
    postnet_dim_range = "{\"dim_range\": {\"0\": {\"min\": [" + str(bs_min) + ", 80, 32], \"max\": [" + str(bs_max) + ", 80, 1664]}}}"
    waveglow_dim_range = "{\"dim_range\": {\"0\": {\"min\": [" + str(bs_min) + ", 80," + str(mel_min) + ",  1], \"max\": [" + str(bs_max) + ", 80, " + str(mel_max) + " ,1]}, \
      \"1\": {\"min\": [" + str(bs_min) + ", 8," + str(z_min) + ", 1], \"max\": [" + str(bs_max) + ", 8," + str(z_max) + ", 1]}}}"

    # Encoder
    if args.encoder != "":
        print("Building Encoder ...")
        encoder_model = build_mm(args.encoder, "encoder", encoder_dim_range, args.shape_mutable, args.precision)
        model_file_path = args.output+"/"+"encoder"+"_"+args.precision+"_"+str(bs_min)+"_"+str(bs_max)+".graph"
        assert encoder_model is not None
        # 将模型序列化为离线文件
        assert encoder_model.serialize_to_file(model_file_path).ok()
    else:
        print("Failed to build mm from", args.encoder)
        sys.exit()
    
    #Decoder
    if args.decoder != "":
        print("Building Decoder ...")
        decoder_model = build_mm(args.decoder, "decoder", decoder_dim_range, args.shape_mutable, args.precision)
        model_file_path = args.output+"/"+"decoder"+"_"+args.precision+"_"+str(bs_min)+"_"+str(bs_max)+".graph"
        assert decoder_model is not None
        # 将模型序列化为离线文件
        assert decoder_model.serialize_to_file(model_file_path).ok()
    else:
        print("Failed to build mm from", args.decoder)
        sys.exit()

    #Postnet
    if args.postnet != "":
        print("Building Postnet ...")
        postnet_model = build_mm(args.postnet, "postnet", postnet_dim_range, args.shape_mutable, args.precision)
        model_file_path = args.output+"/"+"postnet"+"_"+args.precision+"_"+str(bs_min)+"_"+str(bs_max)+".graph"
        assert postnet_model is not None
        # 将模型序列化为离线文件
        assert postnet_model.serialize_to_file(model_file_path).ok()
    else:
        print("Failed to build mm from", args.postnet)
        sys.exit()
    
    ##WaveGlow
    if args.waveglow != "":
        print("Building WaveGlow ...")
        waveglow_model = build_mm(args.waveglow, "waveglow", waveglow_dim_range, args.shape_mutable, args.precision)
        model_file_path = args.output+"/"+"waveglow"+"_"+args.precision+"_"+str(bs_min)+"_"+str(bs_max)+".graph"
        assert waveglow_model is not None
        # 将模型序列化为离线文件
        assert waveglow_model.serialize_to_file(model_file_path).ok()
    else:
        print("Failed to build mm from", args.waveglow)
        sys.exit()

if __name__ == '__main__':
    main()
