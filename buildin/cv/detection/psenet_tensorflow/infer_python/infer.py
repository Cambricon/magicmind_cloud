# -*- coding:utf-8 -*-
import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from pse import pse
import argparse
import json

import magicmind.python.runtime as mm

class MagicmindModel(object):
    def __init__(self, model_path, device_id=0):
        self.model = mm.Model()
        self.model.deserialize_from_file(model_path)
        self.dev = mm.Device()
        self.dev.id = device_id
        assert self.dev.active().ok()
        self.engine = self.model.create_i_engine()
        self.context = self.engine.create_i_context()
        self.queue = self.dev.create_queue()
        self.inputs = self.context.create_inputs()
        self.outputs = []
    
    def run_infer(self, data):
        self.inputs[0].from_numpy(data)
        assert self.context.enqueue(self.inputs, self.outputs, self.queue).ok()
        assert self.queue.sync().ok()
        res = self.outputs[0].asnumpy()
        self.outputs = []
        return res

def get_images(test_data_path,nums):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    # print('Find {} images'.format(len(sfiles)))
    return files[:nums]


def resize_image(im, max_side_len=1200):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(seg_maps, timer, image_w, image_h, min_area_thresh=10, seg_map_thresh=0.9, ratio = 1):
    '''
    restore text boxes from score map and geo map
    :param seg_maps:
    :param timer:
    :param min_area_thresh:
    :param seg_map_thresh: threshhold for seg map
    :param ratio: compute each seg map thresh
    :return:
    '''
    if len(seg_maps.shape) == 4:
        seg_maps = seg_maps[0, :, :, ]
    #get kernals, sequence: 0->n, max -> min
    kernals = []
    one = np.ones_like(seg_maps[..., 0], dtype=np.uint8)
    zero = np.zeros_like(seg_maps[..., 0], dtype=np.uint8)
    thresh = seg_map_thresh
    for i in range(seg_maps.shape[-1]-1, -1, -1):
        kernal = np.where(seg_maps[..., i]>thresh, one, zero)
        kernals.append(kernal)
        thresh = seg_map_thresh*ratio
    start = time.time()
    mask_res, label_values = pse(kernals, min_area_thresh)
    timer['pse'] = time.time()-start
    mask_res = np.array(mask_res)
    mask_res_resized = cv2.resize(mask_res, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
    boxes = []
    for label_value in label_values:
        #(y,x)
        points = np.argwhere(mask_res_resized==label_value)
        points = points[:, (1,0)]
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        boxes.append(box)

    return np.array(boxes), kernals, timer

def show_score_geo(color_im, kernels, im_res):
    fig = plt.figure()
    cmap = plt.cm.hot
    #
    ax = fig.add_subplot(241)
    im = kernels[0]*255
    ax.imshow(im)

    ax = fig.add_subplot(242)
    im = kernels[1]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(243)
    im = kernels[2]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(244)
    im = kernels[3]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(245)
    im = kernels[4]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(246)
    im = kernels[5]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(247)
    im = color_im
    ax.imshow(im)

    ax = fig.add_subplot(248)
    im = im_res
    ax.imshow(im)

    fig.show()

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--magicmind_model', type=str, default='../data/', help='saved .mm model name')
    parser.add_argument('--image_dir', type=str, default='./icdar2015/images/', help='test data path')
    parser.add_argument('--image_num', type=int, default=5, help='img num, default:10')
    parser.add_argument('--device_id', type=int, default=0, help='device id')
    parser.add_argument('--img_size', type=list, default=[704, 1216], help='inference size (pixels)')
    parser.add_argument('--batch_size', type=int, default=1, help='inference batch, default:1') 
    parser.add_argument('--save_img', type=bool, default=False, help='save images, please set save path') 
    parser.add_argument('--save_json', type=bool, default=False, help='save results in json,please set save path') 
    parser.add_argument('--json_path', type=str, default=False, help='save predict results') 
    parser.add_argument('--output_dir', type=str, default='./results/', help='test data path')
    args = parser.parse_args() 

    model = MagicmindModel(args.magicmind_model, device_id=args.device_id)  
    res_dict = dict()     
    im_fn_list = get_images(args.image_dir, args.image_num)
    for im_fn in im_fn_list:
        im = cv2.imread(im_fn)[:, :, ::-1]

        start_time = time.time()
        im_resized, (ratio_h, ratio_w) = resize_image(im)
        h, w, _ = im_resized.shape
        im_resized = im_resized[np.newaxis,:]
        timer = {'net': 0, 'pse': 0}
        start = time.time()
        seg_maps = model.run_infer(im_resized)
        timer['net'] = time.time() - start
        boxes, kernels, timer = detect(seg_maps=seg_maps, timer=timer, image_w=w, image_h=h)

        if boxes is not None:
            boxes = boxes.reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
            h, w, _ = im.shape
            boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w)
            boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h)

        duration = time.time() - start_time
        # save to file
        if boxes is not None:
            num =0
            res = []
            for i in range(len(boxes)):
                # to avoid submitting errors
                box = boxes[i]
                box = box.astype(np.int32)
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                    continue
                num += 1                
                cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=2)
                temp = {"points": [[float(box[0, 0]), float(box[0, 1])],
                                    [float(box[1, 0]), float(box[1, 1])],
                                    [float(box[2, 0]), float(box[2, 1])], 
                                    [float(box[3, 0]), float(box[3, 1])]]}
                res.append(temp)
            image_name = im_fn.split('/')[-1][:-4]
            res_dict[image_name] = res
        if args.save_img:
            img_path = os.path.join(args.output_dir, os.path.basename(im_fn))
            cv2.imwrite(img_path, im[:, :, ::-1])
    if args.save_json:
        with open(args.json_path, 'w+') as f:
            json.dump(res_dict, f)
if __name__ == '__main__':
    main()
