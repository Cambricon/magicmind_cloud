import torch
from torchvision.datasets import CIFAR100
import sys
import os

if __name__=="__main__":
    DATASETS_PATH = sys.argv[1]
    # download dataset
    CIFAR100(root=os.path.expanduser(DATASETS_PATH), download=True, train=False)

    
    