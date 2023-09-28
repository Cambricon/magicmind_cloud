import magicmind.python.runtime as mm
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import sys
import os

sys.path.append("../../../")
from utils import Record

from mm_runner import MMRunner
from logger import Logger

log = Logger()
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


def imagenet_dataset(val_txt, image_file_path, count=-1):
    with open(val_txt, "r") as f:
        lines = f.readlines()
    log.info("%d pictures will be read." % len(lines))
    if len(lines) < count:
        log.info("infer pictures less than {}".format(count))
        return
    current_count = 0
    for line in lines:
        image_name, label = line.split(" ")
        image_path = os.path.join(image_file_path, image_name)
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        yield img, label.strip()
        current_count += 1
        if current_count >= count and count != -1:
            break


def pre_process(input_image, transpose):
    resize_h, resize_w = (256, 256)
    crop_h, crop_w = (IMAGE_HEIGHT, IMAGE_WIDTH)
    # cambricon-note: no need to minus mean and div vars here.
    # + because minus mean and div vars had been done
    # + by `insert_bn_before_firstnode` when generating mm model
    # + through passing means and vars to gen_model.py

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)

    # transforms.ToTensor(),
    # normalize,
    preprocess = transforms.Compose(
        [
            transforms.Resize(resize_h),
            transforms.CenterCrop(crop_h),
        ]
    )
    input_tensor = preprocess(input_image)
    # input_numpy = input_tensor.numpy()
    # if transpose:
    #    input_numpy = np.transpose(input_numpy, (1, 2, 0))
    # return input_numpy

    input_numpy = np.asarray(input_tensor)
    return input_numpy


def load_name(imagenet_label_path):
    name_map = {}
    with open(imagenet_label_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        idx = line.split(" ")[0]
        name = " ".join(line.split(" ")[1:])
        name_map[int(idx)] = name.strip()
    return name_map


# -------- arg parse ------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--device_id", dest="device_id", type=int, default=0, help="device_id"
)
parser.add_argument(
    "--magicmind_model",
    dest="magicmind_model",
    type=str,
    default="../data/models/mm_model",
    help="save mm model to this path",
)
parser.add_argument(
    "--batch_size",
    dest="batch_size",
    type=int,
    default=1,
    help="batch_size used for infer",
)
parser.add_argument(
    "--image_dir",
    dest="image_dir",
    type=str,
    default="/path/to/modelzoo/datasets/imageNet2012/",
    help="imagenet val datasets",
)
parser.add_argument(
    "--image_num", dest="image_num", type=int, default=10, help="image number"
)
parser.add_argument(
    "--name_file",
    dest="name_file",
    type=str,
    default="datasets/imagenet/name.txt",
    help="imagenet name txt",
)
parser.add_argument(
    "--label_file",
    dest="label_file",
    type=str,
    default="/path/to/modelzoo/datasets/imageNet2012/labels.txt",
    help="imagenet val label txt",
)
parser.add_argument(
    "--result_file",
    dest="result_file",
    type=str,
    default="../data/output/infer_result.txt",
    help="result_file",
)
parser.add_argument(
    "--result_label_file",
    dest="result_label_file",
    type=str,
    default="../data/output/eval_labels.txt",
    help="result_label_file",
)
parser.add_argument(
    "--result_top1_file",
    dest="result_top1_file",
    type=str,
    default="../data/output/eval_result_1.txt",
    help="result_top1_file",
)
parser.add_argument(
    "--result_top5_file",
    dest="result_top5_file",
    type=str,
    default="../data/output/eval_result_5.txt",
    help="result_top5_file",
)

# -------- main ------------
if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        log.error(args.magicmind_model + " does not exist.")
        exit()

    # model 定义
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

    # cambricon-note:the default dtype is float32
    dataset = imagenet_dataset(
        val_txt=args.label_file, image_file_path=args.image_dir, count=image_num
    )
    rem_img_num = image_num % batch_size
    img_idx = 0
    batch_counter = 0
    imgs = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    labels = np.empty([batch_size])

    for img, label in tqdm(dataset, total=args.image_num):
        data = pre_process(img, transpose=True)
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
