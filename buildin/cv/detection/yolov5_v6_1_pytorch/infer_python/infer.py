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
from utils import Record

from mm_runner import MMRunner
from logger import Logger

log = Logger()

parser = argparse.ArgumentParser()
parser.add_argument("--device_id", "--device_id", type=int, default=0, help="device_id")
parser.add_argument(
    "--magicmind_model",
    "--magicmind_model",
    type=str,
    default="../data/models/yolov5_pytorch_model_qint8_mixed_float16_true_1",
)
parser.add_argument(
    "--image_dir",
    "--image_dir",
    type=str,
    default="../../../../datasets/coco/val2017",
    help="coco val datasets",
)
parser.add_argument(
    "--image_num", "--image_num", type=int, default=10, help="image number"
)
parser.add_argument(
    "--file_list",
    "--file_list",
    type=str,
    default="coco_file_list_5000.txt",
    help="coco file list",
)
parser.add_argument("--label_path", "--label_path", type=str, default="coco.names")
parser.add_argument(
    "--output_dir", "--output_dir", type=str, default="../data/images/output"
)
parser.add_argument(
    "--imgsz", "--imgsz", default=640, type=int, help="inference size (pixels)"
)
parser.add_argument(
    "--batch_size", "--batch_size", default=8, type=int, help="inference size (pixels)"
)
parser.add_argument("--save_img", "--save_img", type=bool, default=False)

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        log.error(args.magicmind_model + " does not exist.")
        exit()

    # model 定义
    model = MMRunner(mm_file=args.magicmind_model, device_id=args.device_id)

    batch_size = args.batch_size
    image_num = args.image_num
    imgsz = args.imgsz

    img_names = [None for i in range(batch_size)]
    imgs = np.empty([batch_size, imgsz, imgsz, 3], dtype=np.uint8)
    show_imgs = [
        None for i in range(batch_size)
    ]  # np.empty([batch_size,imgsz,imgsz,3],dtype=np.uint8)
    ratios = np.empty([batch_size])

    if isinstance(args.imgsz, int):
        img_size = (imgsz, imgsz)
        dataset = coco_dataset(
            file_list_txt=args.file_list, image_dir=args.image_dir, count=args.image_num
        )
        rem_img_num = image_num % batch_size
        img_idx = 0
        batch_counter = 0
        log.info("Start run ...")
        from tqdm import tqdm

        for img, img_path in tqdm(dataset, total=args.image_num):
            infer_batch_size = (
                batch_size if img_idx < (image_num - rem_img_num) else rem_img_num
            )
            img_names[batch_counter % infer_batch_size] = os.path.splitext(
                img_path.split("/")[-1]
            )[0]

            # pre-process
            show_imgs[batch_counter % infer_batch_size] = img
            img, ratio = letterbox(img, img_size)
            img = img[:, :, ::-1]  # BGR to RGB
            imgs[batch_counter % infer_batch_size, :, :, :] = img
            ratios[batch_counter % infer_batch_size] = ratio

            batch_counter += 1
            img_idx += 1

            if batch_counter % infer_batch_size == 0:
                batch_counter = 0
                inputs = [imgs]

                # inference
                outputs = model(inputs)

                # post-process
                preds = torch.from_numpy(outputs[0])
                detection_nums = torch.from_numpy(outputs[1])
                for pred_idx in range(infer_batch_size):
                    show_img = np.array(show_imgs[pred_idx])
                    pred = preds[pred_idx]
                    detection_num = detection_nums[pred_idx]
                    reshape_value = torch.reshape(pred, (-1, 1))
                    src_h, src_w = show_img.shape[0], show_img.shape[1]
                    scale_w = ratios[pred_idx] * src_w
                    scale_h = ratios[pred_idx] * src_h
                    record = Record(
                        args.output_dir + "/" + img_names[pred_idx] + ".txt"
                    )
                    name_dict = np.loadtxt(args.label_path, dtype="str", delimiter="\n")
                    for k in range(detection_num):
                        class_id = int(reshape_value[k * 7 + 1])
                        score = float(reshape_value[k * 7 + 2])
                        xmin = max(0, min(reshape_value[k * 7 + 3], img_size[1]))
                        xmax = max(0, min(reshape_value[k * 7 + 5], img_size[1]))
                        ymin = max(0, min(reshape_value[k * 7 + 4], img_size[0]))
                        ymax = max(0, min(reshape_value[k * 7 + 6], img_size[0]))
                        xmin = (xmin - (img_size[1] - scale_w) / 2) / ratios[pred_idx]
                        xmax = (xmax - (img_size[1] - scale_w) / 2) / ratios[pred_idx]
                        ymin = (ymin - (img_size[0] - scale_h) / 2) / ratios[pred_idx]
                        ymax = (ymax - (img_size[0] - scale_h) / 2) / ratios[pred_idx]
                        xmin = float(max(0, xmin))
                        xmax = float(max(0, xmax))
                        ymin = float(max(0, ymin))
                        ymax = float(max(0, ymax))
                        result = (
                            name_dict[class_id]
                            + ","
                            + str(score)
                            + ","
                            + str(xmin)
                            + ","
                            + str(ymin)
                            + ","
                            + str(xmax)
                            + ","
                            + str(ymax)
                        )
                        record.write(result, False)
                        if args.save_img:
                            cv2.rectangle(
                                show_img,
                                (int(xmin), int(ymin)),
                                (int(xmax), int(ymax)),
                                (0, 255, 0),
                            )
                            text = name_dict[class_id] + ": " + str(score)
                            text_size, _ = cv2.getTextSize(text, 0, 0.5, 1)
                            cv2.putText(
                                show_img,
                                text,
                                (int(xmin), int(ymin) + text_size[1]),
                                0,
                                0.5,
                                (255, 255, 255),
                                1,
                            )
                    if args.save_img:
                        cv2.imwrite(
                            args.output_dir + "/" + img_names[pred_idx] + ".jpg",
                            show_img,
                        )
