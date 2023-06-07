import numpy as np
import cv2
from typing import List
import os
import sys
import glob
import math
import magicmind.python.runtime as mm
from logger import Logger
# 实例化python logger类
log = Logger()

class CalibData(mm.CalibDataInterface):
    def __init__(self, args, shape: mm.Dims, max_samples: int, img_dir):
        super().__init__()
        self.task = args.task
        self.args = args
        if self.task == 'det':
            self.data_paths_ = glob.glob(img_dir + '/*.jpg')
        else :
            self.data_paths_ = glob.glob(img_dir + '/*.png')
        self.shape_ = shape
        self.batch_size = self.shape_.GetDimValue(0)
        self.max_samples_ = max_samples
        self.cur_sample_ = None
        self.cur_data_index_ = 0
 
    def get_shape(self):
        return self.shape_
 
    def get_data_type(self):
        return mm.DataType.FLOAT32
 
    def get_sample(self):
        return self.cur_sample_

    def preprocess_images(self,data_begin,data_end):
        imgs = []
        for i in range(data_begin,data_end):
            img_path = self.data_paths_[i]
            img = cv2.imread(img_path)
            if self.task == 'det':
                data = {'image': img}
                img, shape_list = db_preprocess(self.args.det_limit_side_len, self.args.det_limit_type, data)
            elif self.task =='rec':
                log.info("if you need rec int8 calib, please contact us")
                log.err("surpport use force_float16 or force_float32")
                # img = rec_preprocess()
            else:
                log.info("if you need cls int8 calib, please contact us")
                log.err("surpport use force_float16 or force_float32")
                # img = cls_preprocess()
            imgs.append(np.ascontiguousarray(img)[np.newaxis,:])
        return np.ascontiguousarray(np.concatenate(tuple(imgs), axis=0).astype(dtype = np.float32))
      
    def next(self):
        batch_size = self.batch_size
        data_begin = self.cur_data_index_
        data_end = data_begin + batch_size
        if data_end > self.max_samples_:
            return mm.Status(mm.Code.OUT_OF_RANGE, "Data end reached")
        self.cur_sample_ = self.preprocess_images(data_begin,data_end)
        self.cur_data_index_ = data_end
        return mm.Status.OK()
 
    def reset(self):
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        return mm.Status.OK()


def db_preprocess(det_limit_side_len, det_limit_type, data):
    std=[0.229, 0.224, 0.225]
    mean=[0.485, 0.456, 0.406]
    scale=1.0 / 255.0
    order='hwc'
    scale = np.float32(scale if scale is not None else 1.0 / 255.0)
    mean = mean if mean is not None else [0.485, 0.456, 0.406]
    std = std if std is not None else [0.229, 0.224, 0.225]
    shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
    mean = np.array(mean).reshape(shape).astype('float32')
    std = np.array(std).reshape(shape).astype('float32')
 
    img = data['image']
    src_h, src_w, _ = img.shape
    limit_side_len = det_limit_side_len
    h, w, c = img.shape
    # limit the max side
    if det_limit_type == 'max':
        if max(h,w) > limit_side_len:
            if h > w:
                ratio = float(limit_side_len)/h
            else:
                ratio = float(limit_side_len)/w
        else:
            ratio = 1
    elif det_limit_type == 'min':
        if min(h,w) < limit_side_len:
            if h < w:
                ratio = float(limit_side_len)/h
            else:
                ratio = float(limit_side_len)/w
        else:
            ratio = 1
    elif det_limit_type == 'resize_long':
            ratio = float(limit_side_len) / max(h, w)
    else:
        raise Exception('not support limit type, image ')
    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)

    try:
        if int(resize_w) <= 0 or int(resize_h) <= 0:
            return None, (None, None)
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
    except:
        print(img.shape, resize_w, resize_h)
        sys.exit(0)

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)  
    data['image'] = img
    data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])

    #nomalize image
    from PIL import Image
    if isinstance(img, Image.Image):
        img = np.array(img)
    assert isinstance(img,np.ndarray), "invalid input 'img' in NormalizeImage"
    data['image'] = (img.astype('float32') * scale - mean) / std

    #convert hwc image to chw image
    img = data['image']
    if isinstance(img, Image.Image):
        img = np.array(img)
    data['image'] = img.transpose((2, 0, 1))
    
    #Keep Keys
    keep_keys=['image', 'shape']
    data_list = []
    for key in keep_keys:
        data_list.append(data[key])
    return data_list

def rec_preprocess(rec_image_shape, limited_min_width, limited_max_width, beg_img_no,img_crop_list,indices,img_num, batch_num): 
    end_img_no = min(img_num, beg_img_no + batch_num)
    norm_img_batch = []

    #cambricon: ori cmd copied from predict_rec.py
    max_wh_ratio = 0
    for ino in range(beg_img_no, end_img_no):
        h, w = img_crop_list[indices[ino]].shape[0:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
    for ino in range(beg_img_no, end_img_no):
        norm_img = resize_norm_img_rec(rec_image_shape,limited_min_width, limited_max_width,img_crop_list[indices[ino]],max_wh_ratio)
        norm_img = norm_img[np.newaxis, :]
        norm_img_batch.append(norm_img)
    norm_img_batch = np.concatenate(norm_img_batch)
    norm_img_batch = norm_img_batch.copy()
    return norm_img_batch

def cls_preprocess(cls_image_shape, limited_min_width, limited_max_width, beg_img_no,img_crop_list,indices,img_num, batch_num):
    end_img_no = min(img_num, beg_img_no + batch_num)
    norm_img_batch = []
    max_wh_ratio = 0
    for ino in range(beg_img_no, end_img_no):
        h, w = img_crop_list[indices[ino]].shape[0:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
    for ino in range(beg_img_no, end_img_no):
        norm_img = resize_norm_img_cls(cls_image_shape,limited_min_width, limited_max_width,img_crop_list[indices[ino]],max_wh_ratio)
        norm_img = norm_img[np.newaxis, :]
        norm_img_batch.append(norm_img)
    norm_img_batch = np.concatenate(norm_img_batch)
    norm_img_batch = norm_img_batch.copy()
    return norm_img_batch

def resize_norm_img_cls(cls_image_shape,limited_min_width, limited_max_width, img, max_wh_ratio):
    cls_image_shape = [int(v) for v in cls_image_shape.split(",")]
    imgC, imgH, imgW = cls_image_shape
    h = img.shape[0]
    w = img.shape[1]
    ratio = w / float(h)
    imgW = max(min(imgW,limited_max_width),limited_min_width)
    ratio_imgH = math.ceil(imgH * ratio)
    ratio_imgH = max(ratio_imgH, limited_min_width)
    if ratio_imgH > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if cls_image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2,0,1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC,imgH,imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im

def resize_norm_img_rec(rec_image_shape,limited_min_width, limited_max_width, img, max_wh_ratio):
    rec_image_shape = [int(v) for v in rec_image_shape.split(",")]
    # 默认输入尺寸
    imgC, imgH, imgW = rec_image_shape
    assert imgC == img.shape[2]
    max_wh_ratio = max(max_wh_ratio, imgW / imgH)
    imgW = int((imgH * max_wh_ratio))
    imgW = max(min(imgW, limited_max_width),limited_min_width)
    h, w = img.shape[:2]
    ratio = w / float(h)
    #按比例缩放
    ratio_imgH = math.ceil(imgH * ratio)
    ratio_imgH = max(ratio_imgH, limited_min_width)
    if ratio_imgH > imgW:
        # 如大于默认宽度，则宽度为imgW
        resized_w = imgW
    else:
        #如小于默认宽度则以图片真实宽度为准
        resized_w = int(ratio_imgH)
    #缩放
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    #归一化
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    #对宽度不足的位置，补0
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


