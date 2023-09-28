import torch
from torchvision.datasets import CIFAR100
import sys
import os

if __name__=="__main__":
    CIFAR100_DATASETS_PATH = sys.argv[1]
    # download dataset
    CIFAR100(root=os.path.expanduser(CIFAR100_DATASETS_PATH), download=True, train=False)

    
    