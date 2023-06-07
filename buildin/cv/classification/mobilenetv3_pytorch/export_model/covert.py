import sys
import os
import torch
sys.path.append('pytorch-mobilenet-v3-master')
from mobilenetv3 import mobilenetv3

if __name__ == "__main__":
    prj_path = sys.argv[1]

    pth_file = os.path.join(prj_path,"data/models/mobilenetv3_small_67.4.pth.tar")
    pytorch_net = mobilenetv3(mode="small")
    pytorch_net.load_state_dict(torch.load(pth_file, map_location='cpu'))
    pytorch_net.eval()
    # jit.trace.save
    INPUT_SIZE = (224, 224) # h, w
    TRACED_PT = os.path.join(prj_path,"data/models/mobilenet-v3_small.torchscript.pt")
    traced_model = torch.jit.trace(pytorch_net, torch.rand(32, 3, INPUT_SIZE[0], INPUT_SIZE[1]))
    torch.jit.save(traced_model, TRACED_PT)