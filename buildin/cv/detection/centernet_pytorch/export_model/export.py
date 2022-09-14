import sys
sys.path.append('CenterNet-2b7692c377c6686fb35e473dac2de6105eed62c6/src/')
import argparse
import torch
import _init_paths
from models.networks.dlav0 import DLASeg as CenterNet
from models.model import load_model
parser = argparse.ArgumentParser()
parser.add_argument("--model_weight", dest = 'model_weight', type=str,default="../data/models/")
parser.add_argument('--input_width', dest = 'input_width', default = 512, type = int, help = 'model input width')
parser.add_argument('--input_height', dest = 'input_height', default = 512, type = int, help = 'model input height')
parser.add_argument('--batch_size', dest = 'batch_size', default = 1, type = int, help = 'model input batch')
parser.add_argument("--traced_pt", dest = 'traced_pt', type=str, default = "../data/models/")

args = parser.parse_args()
# 载入权重
heads = {'hm' : 80, 'wh' : 2, 'reg' :2}
model = CenterNet('dla34', heads, pretrained=True, down_ratio=4, head_conv=256)
model = load_model(model, args.model_weight)
model.eval()

# jit.trace
traced_model = torch.jit.trace(model, torch.rand(args.batch_size, 3, args.input_height, args.input_width))
traced_model.save(args.traced_pt)
