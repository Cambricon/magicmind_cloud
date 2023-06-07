import numpy as np
from magicmind.python.common.types import get_numpy_dtype_by_datatype
from typing import List
import magicmind.python.runtime as mm
from torchvision import transforms, datasets
import os
import glob
import math
import torch
from mmcv import Config, DictAction
from mmaction.datasets import build_dataloader, build_dataset

def build_data_loader(config_file):
    cfg = Config.fromfile(config_file)
    cfg.merge_from_dict({})
    cfg.data.test.test_mode = True
    cfg.setdefault('module_hooks', [])
    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=False,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)
    return data_loader


def preprocess(batch_size,config_file):
    loaders = build_data_loader(config_file)
    for data in loaders:
        np_inputs = data['imgs'].cpu().numpy()
        break
    return np_inputs
    

class CalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, max_samples: int,config_file:str):
        super().__init__()
        self.shape_ = shape
        self.max_samples_ = max_samples
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        self.config_file = config_file
 
    def get_shape(self):
        return self.shape_
 
    def get_data_type(self):
        return mm.DataType.FLOAT32
 
    def get_sample(self):
        return self.cur_sample_
     
    def preprocess_images(self, data_begin: int, data_end: int) -> np.ndarray:
        after_preprocess = preprocess(self.max_samples_,self.config_file)
        return np.ascontiguousarray(after_preprocess)
 
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