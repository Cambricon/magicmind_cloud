import magicmind.python.runtime as mm
import argparse
import numpy as np
import paddle
import paddle.vision.transforms as T
import cv2
import torch
import math
import sys
import os
import logging
from  paddle_utils import multiclass_nms
from metrics import Metric, COCOMetric

transform = T.ToTensor()
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        data_format='CHW')
 
parser = argparse.ArgumentParser()
parser.add_argument("--device_id", "--device_id", type = int, default = 0, help = "device_id")
parser.add_argument("--magicmind_model", "--magicmind_model",type=str,default="../data/models/yolov5_pytorch_model_qint8_mixed_float16_true_1")
parser.add_argument("--image_dir", "--image_dir",  type=str, default="../../../../datasets/coco/val2017", help="coco val datasets")
parser.add_argument("--image_num", "--image_num",  type=int, default=10, help="image number")
parser.add_argument("--file_list", "--file_list",  type=str, default="../../../../datasets/coco/file_list_5000.txt", help="coco file list")
parser.add_argument("--label_path", "--label_path", type=str, default="../../../../datasets/coco/coco.names")
parser.add_argument("--output_dir", "--output_dir", type=str, default="../data/images/output")
parser.add_argument('--input_width', dest = 'input_width', default = 608, type = int, help = 'model input width')
parser.add_argument('--input_height', dest = 'input_height', default = 608, type = int, help = 'model input height')
parser.add_argument('--batch', dest = 'batch', default = 1, type = int, help = 'model input batch')
parser.add_argument("--save_img", "--save_img", type=bool, default=False)
 
def coco_dataset(
    file_list_txt="../../../../datasets/coco/file_list_5000.txt",
    image_dir="../../../../datasets/coco/val2017",
    count=-1
):
    with open(file_list_txt, "r") as f:
        lines = f.readlines()
    logging.info("%d pictures will be read." % len(lines))
    current_count = 0
    for line in lines:
        image_name = line.strip()
        image_path = os.path.join(image_dir, image_name)
        img = cv2.imread(image_path)
        yield img, image_path
        current_count += 1
        if current_count >= count and count != -1:
            break
 
def letterbox(img, dst_shape):
    src_h, src_w = img.shape[0], img.shape[1]
    dst_h, dst_w = dst_shape
    ratio = (dst_h / src_h, dst_w / src_w)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (dst_shape[0], dst_shape[1]), interpolation=cv2.INTER_CUBIC)
    img = transform(img)
    img = normalize(img)
    img = paddle.unsqueeze(img, axis=0).numpy()
    return img, ratio
 
if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        print("please generate magicmind model first!!!")
        exit()
    anno_file = os.environ.get('COCO_DATASETS_PATH') + "/annotations/instances_val2017.json"
    clsid2catid = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
    classwise = False
    output_eval = args.output_dir
    bias = 0
    IouType = 'bbox'
    _metrics = [
            COCOMetric(
                anno_file=anno_file,
                clsid2catid=clsid2catid,
                classwise=classwise,
                output_eval=output_eval,
                bias=bias,
                IouType=IouType)
                ]

    model = mm.Model()
    model.deserialize_from_file(args.magicmind_model)
 
    img_size = [args.input_width, args.input_height]
    with mm.System() as mm_sys:
        dev_count = mm_sys.device_count()
        print("Device count: ", dev_count)
        assert args.device_id < dev_count
        # 打开MLU设备
        dev = mm.Device()
        dev.id = args.device_id
        assert dev.active().ok()
        # 创建Engine
        econfig = mm.Model.EngineConfig()
        econfig.device_type = "MLU"
        engine = model.create_i_engine(econfig)
        assert engine != None, "Failed to create engine"
        # 创建Context
        context = engine.create_i_context()
        assert context != None
        # 创建MLU任务队列
        queue = dev.create_queue()
        assert queue != None
        # 创建输入tensor, 输出tensor
        mm_inputs = context.create_inputs()
        mm_outputs = []
        inputs = [None] * 3
        nms_kwargs={'score_threshold': 0.01, 'nms_top_k': 1000, 'keep_top_k': 100, 'nms_threshold': 0.45, 'normalized': True, 'nms_eta': 1.0, 'return_index': False, 'return_rois_num': True, 'background_label': 80}
        dataset = coco_dataset(file_list_txt = args.file_list, image_dir = args.image_dir, count = args.image_num)
        data = {}
        print("Start run ...")
        for img, img_path in dataset:
            img_name = os.path.splitext(img_path.split("/")[-1])[0]
            print("Inference img : ", img_name)
            # 准备输入数据
            show_img = img
            img, ratio = letterbox(img, img_size)
            inputs[0] = np.array([[img_size[0], img_size[1]]]).astype(np.float32)
            inputs[1] = img
            inputs[2] = np.array([[ratio[0], ratio[1]]]).astype(np.float32)
            for idx, i in enumerate(inputs):
                mm_inputs[idx].from_numpy(inputs[idx])
                mm_inputs[idx].to(dev)
            # 向MLU下发任务
            assert context.enqueue(mm_inputs, mm_outputs, queue).ok()
            # 等待任务执行完成
            assert queue.sync().ok()
            # 处理输出数据
            bbox = mm_outputs[0].asnumpy()
            score = mm_outputs[1].asnumpy()
            bboxes = paddle.to_tensor(bbox)
            scores = paddle.to_tensor(score)
            bbox_pred, bbox_num, _ = multiclass_nms(bboxes, scores, **nms_kwargs)
            bboxes = bbox_pred.numpy()
            bbox_num = bbox_num.numpy()
            outs ={}
            outs['bbox'] = paddle.to_tensor(bbox_pred)
            outs['bbox_num'] = paddle.to_tensor(bbox_num)
            #outs = self.model(data)
            image_id = img_name.split(".")[0]
            data['im_id'] = paddle.to_tensor([[int(image_id)]])
            for metric in _metrics:
                    metric.update(data, outs)
        for metric in _metrics:
            metric.accumulate()
            metric.log()
        for metric in _metrics:
            metric.reset()

