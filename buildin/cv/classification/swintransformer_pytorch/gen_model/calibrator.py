import numpy as np
import cv2
import magicmind.python.runtime as mm
from typing import List
import os
import glob
from PIL import Image
from torchvision import transforms

def preprocess(input_image, transpose):
    resize_h, resize_w = (232, 232)
    crop_h, crop_w = (224, 224)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    preprocess = transforms.Compose(
        [
            transforms.Resize(resize_h, interpolation=Image.BICUBIC),
            transforms.CenterCrop(crop_h),
            transforms.ToTensor(),
            normalize,
        ]
    )
    input_tensor = preprocess(input_image)
    input_numpy = input_tensor.numpy()
    if transpose:
        input_numpy = np.transpose(input_numpy, (1, 2, 0))
    return input_numpy

class CalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, max_samples: int, img_dir: str):
        super().__init__()
        assert os.path.isdir(img_dir)
        self.data_paths_ = glob.glob(img_dir + '/*.jpg')
        t = glob.glob(img_dir + '/*.JPEG')
        self.data_paths_ += t
        self.shape_ = shape
        self.max_samples_ = min(max_samples, len(self.data_paths_))
        self.cur_sample_ = None
        self.cur_data_index_ = 0

    def get_shape(self):
        return self.shape_

    def get_data_type(self):
        return mm.DataType.FLOAT32

    def get_sample(self):
        return self.cur_sample_
    
    def preprocess_images(self, data_begin: int, data_end: int) -> np.ndarray:
        imgs = []
        sample_data = []
        dst_h, dst_w = self.shape_.GetDimValue(2), self.shape_.GetDimValue(3)
        for i in range(data_begin, data_end):
            img = Image.open(self.data_paths_[i])
            if img.mode != "RGB":
                img = img.convert("RGB")
            data = preprocess(img, transpose=False)
            sample_data.append(np.expand_dims(data, 0))

        return np.ascontiguousarray(sample_data)
    
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



