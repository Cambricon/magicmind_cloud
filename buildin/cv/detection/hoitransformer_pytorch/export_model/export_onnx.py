# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------

import argparse
import random

import torch

from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    # Backbone.
    parser.add_argument('--backbone', choices=['resnet50', 'resnet101'], required=True,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer.
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss.
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # Matcher.
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # Loss coefficients.
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.02, type=float,
                        help="Relative classification weight of the no-object class")

    # Dataset parameters.
    parser.add_argument('--dataset_file', choices=['hico', 'vcoco', 'hoia'], required=True)

    parser.add_argument('--model_path', required=True,
                        help='Path of the model to evaluate.')
    parser.add_argument('--log_dir', default='./',
                        help='path where to save temporary files in test')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)

    # Distributed training parameters.
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Visualization.
    parser.add_argument('--img_sheet', help='File containing image paths.')
    return parser



def load_model(model_path, args):
    checkpoint = torch.load(model_path, map_location='cpu')
    print('epoch:', checkpoint['epoch'])
    model, criterion = build_model(args)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model



def export_onnx(model):

    onnx_model_name = "../data/models/hoitransformer.onnx"
    torch.onnx.export(model,         # model being run
         torch.randn(1,3,672,896),       # model input (or a tuple for multiple inputs)
         onnx_model_name,       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=11,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size', 1: 'channels', 2: 'h', 3: 'w'},    # variable length axes
             'modelOutput' : {0 : 'batch_size', 1: 'h', 2: 'w'}})
    return onnx_model_name

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    model = load_model(model_path=args.model_path, args=args)
    onnx_model_name = export_onnx(model)
    print('done')
    print("onnx model name: ", onnx_model_name)


if __name__ == '__main__':
    main()
