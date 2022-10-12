import os
from skimage import io, transform
import torch
import torchvision
import torch
import sys
sys.path.append("./U-2-Net")
from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
#import torch_mlu.core.mlu_model as ct

def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2net
    MODEL_PATH = os.getenv("MODEL_PATH")
    model_dir = os.path.join(MODEL_PATH, model_name + '.pth')

    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    trace_input = torch.randn(1, 3, 320, 320, dtype = torch.float)
    trace_model = torch.jit.trace(net, (trace_input))
    torch.jit.save(trace_model, MODEL_PATH + os.sep + model_name + '.pt')
    exit()

if __name__ == "__main__":
    main()
