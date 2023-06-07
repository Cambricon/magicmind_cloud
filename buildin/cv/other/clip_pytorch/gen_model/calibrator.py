import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
import clip
from torchvision.datasets import CIFAR100
import torch
import numpy as np
import glob
import os 
import cv2
import os

class CalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, index: int, max_samples: int):
        super().__init__()
        self.shape_ = shape
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        self.model, self.preprocess = clip.load("ViT-B/32", device="cpu")
        self.cifar100 = CIFAR100(root=os.path.expanduser(os.environ.get('CIFAR100_DATASETS_PATH')), download=True, train=False,transform=self.preprocess)
        self.max_samples_ = max_samples
        self.text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.cifar100.classes]).to("cpu")
        self.index = index

    def get_shape(self):
        if (self.index == 0):
          return self.shape_ 
        else:
          return mm.Dims((100, 77))

    def get_data_type(self):
        if (self.index == 0):
          return mm.DataType.FLOAT32
        else:
          return mm.DataType.INT32

    def get_sample(self):
        return self.cur_sample_
    
    def preprocess_images(self, data_begin: int, data_end: int) -> np.ndarray:
        imgs = []
        for i in range(data_begin,data_end):
            image, class_id = self.cifar100[i]
            imgs.append(image.numpy())
        if (self.index == 0):
            return np.array(imgs)
        else:
            return self.text_inputs.numpy().astype(np.int32)

    def next(self):
        batch_size = self.shape_.GetDimValue(0)
        data_begin = self.cur_data_index_
        data_end = data_begin + batch_size
        if data_end > self.max_samples_:
            return mm.Status(mm.Code.OUT_OF_RANGE, "Data end reached")
        self.cur_sample_ = self.preprocess_images(data_begin, data_end)
        self.cur_data_index_ = data_end
        return mm.Status.OK()

    def reset(self):
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        return mm.Status.OK()
