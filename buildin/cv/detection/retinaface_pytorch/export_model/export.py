import sys
import os
PROJ_ROOT_PATH = os.environ.get("PROJ_ROOT_PATH")
sys.path.append('Pytorch_Retinaface')
import torch
from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace

import argparse

cfgs = dict([('mobile0.25', cfg_mnet), ('resnet50', cfg_re50)])

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch_size")
parser.add_argument("--cfg", type=str, default='mobile0.25', help="cfg file, can be \'mobile0.25\' or \'resnet50\'")
parser.add_argument('--weights', type=str, default= str(PROJ_ROOT_PATH) + '/data/models/Resnet50_Final.pth', help='weights path')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[672, 1024], help='image (h, w)')
parser.add_argument("--traced_pt", type=str, default= str(PROJ_ROOT_PATH) + '/data/models/retinaface_traced.pt', help="traced pt file")

# 参考test_widerface.py
def remove_prefix(state_dict, prefix):
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

if __name__ == "__main__":
    args = parser.parse_args()
    cfgs[args.cfg]['pretrain']=False
    model = RetinaFace(cfg=cfgs[args.cfg], phase='test')
    pretrained_dict = torch.load(args.weights, map_location='cpu')
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    model.load_state_dict(pretrained_dict, strict=False)
    model.eval()
    rand_input = torch.randn(args.batch_size, 3, args.imgsz[0], args.imgsz[1]).float()
    traced_model = torch.jit.trace(model, rand_input)
    torch.jit.save(traced_model, args.traced_pt)

