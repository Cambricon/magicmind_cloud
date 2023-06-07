import os 

import torch
import torchvision 

if __name__ == "__main__":
    
    resnet50_model = torchvision.models.resnet50(pretrained=False)
    
    resnet50_model.load_state_dict(torch.load("../data/models/resnet50-0676ba61.pth",map_location = 'cpu'))
    resnet50_model.eval()

    inputs = torch.empty(1,3,224,224)

    torch.jit.save(torch.jit.trace(resnet50_model,inputs),'../data/models/resnet50.pt')
    print("successfully save pt")
