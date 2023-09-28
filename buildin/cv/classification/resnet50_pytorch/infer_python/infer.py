import os
import numpy as np
import magicmind.python.runtime as mm
import argparse
import sys
import time
from tqdm import tqdm
sys.path.append("..")
from gen_model.preprocess import preprocess, imagenet_dataset

from utils import Record
from mm_runner import MMRunner
from logger import Logger
log = Logger()

def load_name(imagenet_label_path):
    name_map = {}
    with open(imagenet_label_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        idx = line.split(" ")[0]
        name = " ".join(line.split(" ")[1:])
        name_map[int(idx)] = name.strip()
    return name_map

parser = argparse.ArgumentParser()
parser.add_argument("--device_id", "--device_id",  type=int, default=0, help="device_id")
parser.add_argument("--magicmind_model", "--magicmind_model", type=str, default="../data/models/renset50_onnx_model", help="save mm model to this path")
parser.add_argument("--image_dir", "--image_dir",  type=str, default="/path/to/modelzoo/datasets/imageNet2012/", help="imagenet val datasets")
parser.add_argument("--image_num", "--image_num",  type=int, default=10, help="image number")
parser.add_argument("--name_file", "--name_file",  type=str, default="datasets/imagenet/name.txt", help="imagenet name txt")
parser.add_argument("--label_file", "--label_file",  type=str, default="/path/to/modelzoo/datasets/imageNet2012/labels.txt", help="imagenet val label txt")
parser.add_argument("--result_file", "--result_file",  type=str, default="../data/images/output/infer_result.txt", help="result_file")
parser.add_argument("--result_label_file", "--result_label_file",  type=str, default="../data/images/output/eval_labels.txt", help="result_label_file")
parser.add_argument("--result_top1_file", "--result_top1_file",  type=str, default="../data/images/output/eval_result_1.txt", help="result_top1_file")
parser.add_argument("--result_top5_file", "--result_top5_file",  type=str, default="../data/images/output/eval_result_5.txt", help="result_top5_file")
parser.add_argument("--batch_size", "--batch_size",  type=int, default=1, help="batch_size")

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        log.info("please generate magicmind model first!!!")
        exit()
        
    # model 定义    
    model = MMRunner(mm_file = args.magicmind_model,device_id = args.device_id)
    batch_size = args.batch_size 
    image_num = args.image_num

    name_map = load_name(args.name_file)
    record = Record(args.result_file)
    result_label = Record(args.result_label_file)
    result_top1 = Record(args.result_top1_file)
    result_top5 = Record(args.result_top5_file)

    dataset = imagenet_dataset(val_txt = args.label_file, image_file_path = args.image_dir, count = args.image_num)
    count = 0
    rem_img_num = image_num % batch_size
    img_idx = 0
    batch_counter = 0
    imgsz =  224
    imgs = np.empty([batch_size,imgsz,imgsz,3],dtype=np.float)
    labels = [ None for i in range(batch_size) ]
    
    for img, label in tqdm(dataset, total=args.image_num):
        # calc infer_batch_size
        infer_batch_size = batch_size if img_idx < (image_num-rem_img_num) else rem_img_num
        # pre-process
        imgs[ batch_counter % infer_batch_size,:,:,:] = preprocess(img, transpose = True, normalization = False)
        labels[ batch_counter % infer_batch_size] = label

        batch_counter += 1
        img_idx += 1

        if batch_counter % infer_batch_size == 0:
            batch_counter = 0 
            inputs = [imgs]
            # inference
            outputs = model(inputs)

            # post-process
            for pred_idx in range(infer_batch_size):
                index = outputs[0][pred_idx].argsort()[::-1]
                record.write("top5 result:", False)
                result_label.write("[%d]: %d"%(count, int(labels[pred_idx])), False)
                result_top1.write("[%d]: %d"%(count, index[0]), False)
                for i in range(5):
                    idx = index[i]
                    name = name_map[idx]
                    record.write("%d [%s]"%(i, name), False)
                    result_top5.write("[%d]: %d"%(count, idx), False)
                count += 1

