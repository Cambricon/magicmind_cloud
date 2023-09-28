import os
from PIL import Image
import numpy as np
import magicmind.python.runtime as mm
import argparse
import sys
import clip
import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tqdm import tqdm
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
  default="../data/models/clip_onnx", 
  help="save mm model to this path"
)
parser.add_argument(
  "--image_dir", 
  dest="image_dir",  
  type=str, 
  default="/path/to/datasets/cifar100/",
   help="cifar100 val datasets"
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
  help="result_top5_file"
)
parser.add_argument(
  "--batch_size", 
  dest="batch_size",  
  type=int, 
  default=1, 
  help="batch_size"
)
parser.add_argument(
    "--image_num", 
    dest="image_num", 
    type=int, 
    default=1000, 
    help="image number"
)


if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        print("please generate magicmind model first!!!")
        exit()
    
    model = MMRunner(mm_file=args.magicmind_model, device_id=args.device_id)

    record = Record(args.result_file)
    result_label = Record(args.result_label_file)
    result_top1 = Record(args.result_top1_file)
    result_top5 = Record(args.result_top5_file)
    clipmodel, preprocess = clip.load('ViT-B/32', "cpu")
    root = os.path.expanduser(args.image_dir)
    test = CIFAR100(root, download=True, train=False, transform=preprocess)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test.classes]).to("cpu")
    count = 0
    log.info("Start run ...")
    batch_size = args.batch_size
    image_num = args.image_num
    rem_img_num = image_num % batch_size
    img_idx = 0
    batch_counter = 0
    inputs=[]
    total_tmp = int(image_num/batch_size)
    if total_tmp*batch_size < image_num:
        total_tmp += 1

    imgs = np.empty([batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH])
    val_loader = DataLoader(test,batch_size=args.batch_size,drop_last=False)
    for images, labels in tqdm(val_loader,total=total_tmp,ncols = 80):
      infer_batch_size = (
            batch_size if img_idx < (image_num - rem_img_num) else rem_img_num
      )
      if img_idx >= image_num:
            break
      for index in range (infer_batch_size):
            image = images[index]
            imgs[batch_counter % infer_batch_size, :, :, :] = image
            batch_counter += 1
            img_idx += 1
      if batch_counter % infer_batch_size == 0:
            inputs.append(imgs)
            inputs.append(text_inputs.numpy())
            outputs = model(inputs)
            logits_per_image = torch.from_numpy(outputs[0])
            logits_per_text = torch.from_numpy(outputs[1])
            probs = logits_per_image.softmax(dim=-1).cpu()
            vals, topk1 = torch.topk(probs,1)
            vals, topk5 = torch.topk(probs,5)
            for pred_idx in range(infer_batch_size):
              record.write("top5 result:", False)
              result_label.write("[%d]: %d"%(count, labels[pred_idx]), False)
              result_top1.write("[%d]: %d"%(count, topk1[pred_idx][0]), False)
              for j in range(5):
                idx = topk5[pred_idx][j]
                name = test.classes[idx]
                record.write("%d [%s]"%(j, name), False)
                result_top5.write("[%d]: %d"%(count, idx), False)
              count += 1
