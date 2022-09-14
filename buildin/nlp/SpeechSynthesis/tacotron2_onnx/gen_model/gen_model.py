import argparse
import json

from magicmind.python.runtime.parser import Parser
from magicmind.python.runtime import Network, Builder, Model
from magicmind.python.runtime import BuilderConfig, ModelKind, Dims

import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from mm_utils import _check_status

# For a single dimension this will return the min, opt, and max size when given
# input of either one or three (comma delimited) values
#   dim="1" or dim=1 returns (1, 1, 1)
#   dim="1,4,5" returns (1, 4, 5)
def parse_dynamic_size(dim):
    split = str(dim).split(',')
    assert len(split) in (1,3) , "Dynamic size input must be either 1 or 3 comma-separated integers"
    ints = [int(i) for i in split]
    
    if len(ints) == 1:
        ints *= 3

    assert ints[0] <= ints[1] <= ints[2]
    return tuple(ints)

def build_mm(model_file, model_name, builder_config, dim_range_string=None):
    network = Network()
    builder = Builder()
    builder_cfg = BuilderConfig()

    builder_cfg.parse_from_string(json.dumps(builder_config.get("builder_config")))

    if dim_range_string:
        builder_cfg.parse_from_string(dim_range_string)
    
    parser = Parser(ModelKind.kOnnx)

    _check_status(parser.parse(network, model_file))

    model = builder.build_model(model_name, network, builder_cfg)

    return model

def parse_args(parser):
    """
      Parse commandline arguments.
    """
    parser.add_argument('-o', '--output', required=True,
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('--json', type=str, required=True,
                        help='builder config file')
    parser.add_argument('--encoder', type=str, default="",
                        help='full path to the Encoder ONNX')
    parser.add_argument('--decoder', type=str, default="",
                        help='full path to the DecoderIter ONNX')
    parser.add_argument('--postnet', type=str, default="",
                        help='full path to the Postnet ONNX')
    parser.add_argument('--waveglow', type=str, default="",
                        help='full path to the WaveGlow ONNX')
    parser.add_argument("--quant_mode", type=str, default="force_float16",
                        help='quant mode')
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

    if args.json:
        with open(args.json,'r',encoding='utf8') as fp:
            builder_config = json.load(fp)
    else:
        print("must be set builder config path")
        sys.exit()

    if args.quant_mode == "force_float16":
        builder_config["builder_config"]['precision_config']['precision_mode'] = "force_float16"

    path=args.output.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)

    bs_min, bs_opt, bs_max = parse_dynamic_size(args.batch_size)
    mel_min, mel_opt, mel_max = parse_dynamic_size(args.mel_size)
    z_min, z_opt, z_max = parse_dynamic_size(args.z_size)

    builder_config["builder_config"]["graph_shape_mutable"] = True

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
        encoder_model = build_mm(args.encoder, "encoder", builder_config,
                                 encoder_dim_range)
        model_file_path = args.output+"/"+"encoder_"+args.quant_mode+"_model"
        _check_status(encoder_model.serialize_to_file(model_file_path))
    else:
        print("Failed to build mm from", args.encoder)
        sys.exit()
    
    #Decoder
    if args.decoder != "":
        print("Building Decoder ...")
        decoder_model = build_mm(args.decoder, "decoder", builder_config,
                                 decoder_dim_range)
        model_file_path = args.output+"/"+"decoder_"+args.quant_mode+"_model"
        _check_status(decoder_model.serialize_to_file(model_file_path))
    else:
        print("Failed to build mm from", args.decoder)
        sys.exit()

    #Postnet
    if args.postnet != "":
        print("Building Postnet ...")
        postnet_model = build_mm(args.postnet, "postnet", builder_config,
                                 postnet_dim_range)
        model_file_path = args.output+"/"+"postnet_"+args.quant_mode+"_model"
        _check_status(postnet_model.serialize_to_file(model_file_path))
    else:
        print("Failed to build mm from", args.postnet)
        sys.exit()
    
    ##WaveGlow
    if args.waveglow != "":
        print("Building WaveGlow ...")
        waveglow_model = build_mm(args.waveglow, "waveglow", builder_config,
                                  waveglow_dim_range)
        model_file_path = args.output+"/"+"waveglow_"+args.quant_mode+"_model"
        _check_status(waveglow_model.serialize_to_file(model_file_path))
    else:
        print("Failed to build mm from", args.waveglow)
        sys.exit()

if __name__ == '__main__':
    main()

