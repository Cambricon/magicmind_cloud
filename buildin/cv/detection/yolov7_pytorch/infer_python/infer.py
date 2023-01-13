import magicmind.python.runtime as mm
import argparse
import numpy as np
import cv2
import torch
import math
import sys
import os
sys.path.append("..")
from gen_model.calibrator import letterbox, coco_dataset
sys.path.append("../../../")
from utils.utils import Record

parser = argparse.ArgumentParser()
parser.add_argument("--device_id", "--device_id", type = int, default = 0, help = "device_id")
parser.add_argument("--magicmind_model", "--magicmind_model",type=str,default="../data/models/yolov7_pytorch_model_qint8_mixed_float16_true_1")
parser.add_argument("--image_dir", "--image_dir",  type=str, default="../../../../datasets/coco/val2017", help="coco val datasets")
parser.add_argument("--image_num", "--image_num",  type=int, default=10, help="image number")
parser.add_argument("--file_list", "--file_list",  type=str, default="coco_file_list_5000.txt", help="coco file list")
parser.add_argument("--label_path", "--label_path", type=str, default="coco.names")
parser.add_argument("--output_dir", "--output_dir", type=str, default="../data/images/output")
parser.add_argument('--imgsz', '--imgsz', default = 640, type = int, help = 'inference size (pixels)')
parser.add_argument("--save_img", "--save_img", type=bool, default=False)

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        print("please generate magicmind model first!!!")
        exit()
    model = mm.Model()
    model.deserialize_from_file(args.magicmind_model)
    
    if isinstance(args.imgsz, int):
        img_size = (args.imgsz, args.imgsz)
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
        inputs = context.create_inputs()
        outputs = []
        
        dataset = coco_dataset(file_list_txt = args.file_list, image_dir = args.image_dir, count = args.image_num)
        print("Start run ...")
        from tqdm import tqdm
        for img, img_path in tqdm(dataset, total=args.image_num):
            img_name = os.path.splitext(img_path.split("/")[-1])[0]
            # 准备输入数据
            show_img = img
            img, ratio = letterbox(img, img_size)
            # BGR to RGB
            img = img[:, :, ::-1]
            img = np.expand_dims(img, 0) # (1, 640, 640, 3)
            inputs[0].from_numpy(img)
            inputs[0].to(dev)
            # 向MLU下发任务
            assert context.enqueue(inputs, outputs, queue).ok()
            # 等待任务执行完成
            assert queue.sync().ok()
            # 处理输出数据
            pred = torch.from_numpy(np.array(outputs[0].asnumpy()))
            detection_num = torch.from_numpy(np.array(outputs[1].asnumpy()))
            reshape_value = torch.reshape(pred, (-1, 1))
            src_h, src_w = show_img.shape[0], show_img.shape[1]
            scale_w = ratio * src_w 
            scale_h = ratio * src_h

            record = Record(args.output_dir + "/" + img_name + '.txt')
            name_dict = np.loadtxt(args.label_path, dtype='str', delimiter='\n')
            for k in range(detection_num):
                class_id = int(reshape_value[k * 7 + 1])
                score = float(reshape_value[k * 7 + 2])
                xmin = max(0, min(reshape_value[k * 7 + 3], img_size[1]))
                xmax = max(0, min(reshape_value[k * 7 + 5], img_size[1]))
                ymin = max(0, min(reshape_value[k * 7 + 4], img_size[0]))
                ymax = max(0, min(reshape_value[k * 7 + 6], img_size[0]))
                xmin = (xmin - (img_size[1] - scale_w) / 2) / ratio
                xmax = (xmax - (img_size[1] - scale_w) / 2) / ratio
                ymin = (ymin - (img_size[0] - scale_h) / 2) / ratio
                ymax = (ymax - (img_size[0] - scale_h) / 2) / ratio
                xmin = float(max(0, xmin))
                xmax = float(max(0, xmax))
                ymin = float(max(0, ymin))
                ymax = float(max(0, ymax))
                result = name_dict[class_id]+"," +str(score)+","+str(xmin)+","+str(ymin)+","+str(xmax)+","+str(ymax)
                record.write(result, False)
                if args.save_img:
                    cv2.rectangle(show_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0))
                    text = name_dict[class_id] + ": " + str(score)
                    text_size, _ = cv2.getTextSize(text, 0, 0.5, 1)
                    cv2.putText(show_img, text, (int(xmin), int(ymin) + text_size[1]), 0, 0.5, (255, 255, 255), 1)
            if args.save_img:
                cv2.imwrite(args.output_dir + "/" + img_name + ".jpg", show_img)
