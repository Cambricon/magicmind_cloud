#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import io
import os

class lmdbDataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None):
        self.root = root

        self.transform = transform
        self.target_transform = target_transform

        imagePathList = self.root + "/annotation.txt"
        with open(imagePathList, "r") as f:
            self.annotation = f.readlines()
        self.annotation.sort()
        self.nSamples = len(self.annotation) - 1
        print(self.nSamples)

    def open_lmdb(self):
        self.env = lmdb.open(
            self.root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        if not self.env:
            print("cannot create lmdb from %s" % (self.root))
            sys.exit(0)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        index += 1

        i = self.annotation[index]
        imagePath = i.split(" ")[0]
        label = i.split(" ")[1]
        if '\n' in label:
            label=label.strip("\n")
        imagePath = os.path.join(self.root + "/mnt/ramdisk/max/90kDICT32px/", imagePath)
        byteImgIO = io.BytesIO()
        byteImg = Image.open(imagePath)
        byteImg.save(byteImgIO, "PNG")
        byteImgIO.seek(0)
        byteImg = byteImgIO.read()
        dataBytesIO = io.BytesIO(byteImg)
        try:
            img = Image.open(dataBytesIO).convert("L")
        except IOError:
            print("Corrupted image for %d" % index)
            return self[index + 1]
        if self.transform is not None:
            img = self.transform(img)

        label_key = "label-%09d" % index
        label = str(label)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return (img, label)

class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):
    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size : (i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size :] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels

