import torch
import torchvision
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights

weights = Swin_T_Weights.IMAGENET1K_V1
model = swin_t(weights = weights)
model.eval()
 
dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
torch.onnx.export(model, dummy_input, "../data/models/swin.onnx",
                  input_names = ['images'],
                  opset_version = 11,
                  output_names = ['output'],
                  dynamic_axes={"images":{0:"batch_size"},'output':{0:"batch_size"}}
                  )
