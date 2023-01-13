import clip
import torch
import sys
import os

if __name__=="__main__":
    PRJ_ROOT_PATH = sys.argv[1]    
    ONNX_MODEL_PATH = os.path.join(PRJ_ROOT_PATH,'data/models/clip.onnx')
    if not os.path.exists(ONNX_MODEL_PATH):
        save_dir = os.path.join(PRJ_ROOT_PATH,'data/models')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        image = torch.randn(1 ,3 , 224, 224)
        text = torch.randint(1, 64, (3,77))
        torch.onnx.export(model,(image,text),ONNX_MODEL_PATH,verbose=True,opset_version=10)
        print("Export torch model to onnx model sucess!")
    

    
    