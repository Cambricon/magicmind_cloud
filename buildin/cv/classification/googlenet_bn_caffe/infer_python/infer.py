import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import magicmind.python.runtime as mm
import argparse
import sys
sys.path.append("..")
from preprocess import preprocess, imagenet_dataset,load_name
sys.path.append("../../../")
from utils import Record

from mm_runner import MMRunner
from logger import Logger

log = Logger()
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

parser = argparse.ArgumentParser()
parser.add_argument(
    "--device_id", 
    dest="device_id",  
    type=int, 
    default=0, 
    help="device_id"
)
parser.add_argument(
    "--magicmind_model", 
    dest="magicmind_model", 
    type=str, 
    default="../data/models/googlenet_bn_caffe", 
    help="save mm model to this path"
)
parser.add_argument(
    "--batch_size", 
    dest="batch_size",  
    type=int, 
    default=1, 
    help="batch_size"
)
parser.add_argument(
    "--image_dir", 
    dest="image_dir",  
    type=str, 
    default="/path/to/modelzoo/datasets/imageNet2012/",
    help="imagenet val datasets"
)
parser.add_argument(
    "--image_num", 
    dest="image_num",  
    type=int, 
    default=10, 
    help="image number"
)
parser.add_argument(
    "--name_file", 
    dest="name_file",
    type=str, 
    default="datasets/imagenet/name.txt", 
    help="imagenet name txt"
)
parser.add_argument(
    "--label_file", 
    dest="label_file",
    type=str, 
    default="/path/to/modelzoo/datasets/imageNet2012/labels.txt",
    help="imagenet val label txt"
)
parser.add_argument(
    "--result_file", 
    dest="result_file",
    type=str, 
    default="../data/images/output/infer_result.txt", 
    help="result_file"
)
parser.add_argument(
    "--result_label_file", 
    dest="result_label_file",  
    type=str, 
    default="../data/images/output/eval_labels.txt",
    help="result_label_file"
)
parser.add_argument(
    "--result_top1_file", 
    dest="result_top1_file", 
    type=str, 
    default="../data/images/output/eval_result_1.txt", 
    help="result_top1_file"
)
parser.add_argument(
    "--result_top5_file", 
    dest="result_top5_file",  
    type=str, 
    default="../data/images/output/eval_result_5.txt", 
    help="result_top5_file")

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        print("please generate magicmind model first!!!")
        exit()
    model = MMRunner(mm_file=args.magicmind_model, device_id=args.device_id)
    name_map = load_name(args.name_file)
    record = Record(args.result_file)
    result_label = Record(args.result_label_file)
    result_top1 = Record(args.result_top1_file)
    result_top5 = Record(args.result_top5_file)

    count = 0
    log.info("Start run ...")
    batch_size = args.batch_size
    image_num = args.image_num

    dataset = imagenet_dataset(
        val_txt = args.label_file, image_file_path = args.image_dir, count = args.image_num
    )
    
    rem_img_num = image_num % batch_size
    img_idx = 0
    batch_counter = 0
    imgs = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    labels = np.empty([batch_size])
    
    for img, label in tqdm(dataset, total=args.image_num):
        data = preprocess(img, transpose=False)
        infer_batch_size = (
            batch_size if img_idx < (image_num - rem_img_num) else rem_img_num
        )
        imgs[batch_counter % infer_batch_size, :, :, :] = data
        labels[batch_counter % infer_batch_size] = label
        batch_counter += 1
        img_idx += 1

        if batch_counter % infer_batch_size == 0:
            batch_counter = 0
            inputs = [imgs]

            # inference
            outputs = model(inputs)

            # post-process
            for pred_idx in range(infer_batch_size):
                pred = outputs[0][pred_idx]
                index = pred.argsort()[::-1]

                record.write("top5 result:", False)
                result_label.write("[%d]: %d" % (count, int(labels[pred_idx])), False)
                result_top1.write("[%d]: %d" % (count, index[0]), False)

                for i in range(5):
                    idx = index[i]
                    name = name_map[idx]
                    record.write("%d [%s]" % (i, name), False)
                    result_top5.write("[%d]: %d" % (count, idx), False)
                count += 1
