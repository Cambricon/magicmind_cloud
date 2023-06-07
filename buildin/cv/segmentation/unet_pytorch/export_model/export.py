import argparse
import logging
import os, sys

import torch
sys.path.append("Pytorch-UNet")
from unet import UNet

MODEL_PATH = os.getenv("MODEL_PATH")

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default= '../data/models/unet_carvana_scale0.5_epoch2.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--output_model', '-o', default= '../data/models/unet_carvana_scale0.5_epoch2_trace.pt', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    tarce_name = args.output_model
    
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    device = torch.device('cpu')
    logging.info(f'Loading model {args.model}')

    net.to(device=device) 
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    net.eval()
    trace_input = torch.randn(1, 3, 640, 959, dtype = torch.float)
    trace_model = torch.jit.trace(net, trace_input)
    torch.jit.save(trace_model, tarce_name)
    logging.info('torch.jit.trace model saved in ' + tarce_name)
