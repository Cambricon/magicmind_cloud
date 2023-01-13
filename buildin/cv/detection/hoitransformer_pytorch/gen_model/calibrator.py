import magicmind.python.runtime as mm
import numpy as np
import glob
import os 
import cv2
from PIL import Image
import torchvision
import torchvision.transforms.functional as F
import random

def resize(image, target, size, max_size=None, image_set='train'):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return h, w
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return oh, ow

    rescale_size = get_size_with_aspect_ratio(image_size=image.size, size=size, max_size=max_size)
    rescaled_image = F.resize(image, rescale_size)

    if target is None:
        return rescaled_image, None
    target = target.copy()
    if image_set in ['test']:
        return rescaled_image, target

    return rescaled_image, target


class ToTensor(object):
    def __call__(self, img, target, image_set='train'):
        return torchvision.transforms.functional.to_tensor(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target, image_set='train'):
        image = torchvision.transforms.functional.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        if image_set in ['test']:
            return image, target
        h, w = image.shape[-2:]
        return image, target



class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None, image_set='train'):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size, image_set)



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, image_set='train'):
        for t in self.transforms:
            image, target = t(image, target, image_set)
        return image, target

def make_hico_transforms(image_set, test_scale=-1):
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if image_set in ['test']:
        if test_scale == -1:
            return Compose([
                normalize,
            ])
        assert 400 <= test_scale <= 800*2, test_scale
        return Compose([
            RandomResize([test_scale], max_size=1333*2),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')



class CalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, max_samples: int, img_dir: str):
        super().__init__()
        assert os.path.isdir(img_dir)
        self.data_paths_ = glob.glob(img_dir + '/*.png')
        #t = glob.glob(img_dir + '/*.JPEG')
        #self.data_paths_ += t
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
        image_set = "test"
        test_scale = 672
        for i in range(data_begin, data_end):
            img = cv2.imread(self.data_paths_[i], cv2.IMREAD_COLOR)
            
            img = Image.fromarray(img[:, :, ::-1]).convert('RGB')
            transforms = make_hico_transforms(image_set, test_scale)
            target = None
            img, target = transforms(img, target, image_set)
            img = img.numpy()
            imgs.append(np.ascontiguousarray(img)[np.newaxis,:])
        # batch
        return np.ascontiguousarray(np.concatenate(tuple(imgs), axis=0))

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
