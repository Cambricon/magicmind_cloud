import os
from PIL import Image
import numpy as np
import magicmind.python.runtime as mm
import argparse
import sys
sys.path.append("..")
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import logging
sys.path.append("../../../")
from utils.utils import Record

def imagenet_dataset(
    val_txt,
    image_file_path,
    count=-1
):
    with open(val_txt, "r") as f:
        lines = f.readlines()
    logging.info("%d pictures will be read." % len(lines))
    if len(lines) < count:
        print("infer pictures less than {}".format(count))
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
       


def preprocess(input_image, transpose):
    resize_h, resize_w = (232, 232)
    crop_h, crop_w = (224, 224)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    preprocess = transforms.Compose(
        [
            transforms.Resize(resize_h, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_h),
            transforms.ToTensor(),
            normalize,
        ]
    )
    input_tensor = preprocess(input_image)
    input_numpy = input_tensor.numpy()
    if transpose:
        input_numpy = np.transpose(input_numpy, (1, 2, 0))
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

parser = argparse.ArgumentParser()
parser.add_argument("--device_id", "--device_id",  type=int, default=0, help="device_id")
parser.add_argument("--magicmind_model", "--magicmind_model", type=str, default="../data/models/swin_onnx_model", help="save mm model to this path")
parser.add_argument("--image_dir", "--image_dir",  type=str, default="/nfsdata/datasets/imageNet2012/", help="imagenet val datasets")
parser.add_argument("--image_num", "--image_num",  type=int, default=10, help="image number")
parser.add_argument("--name_file", "--name_file",  type=str, default="datasets/imagenet/name.txt", help="imagenet name txt")
parser.add_argument("--label_file", "--label_file",  type=str, default="/nfsdata/datasets/imageNet2012/labels.txt", help="imagenet val label txt")
parser.add_argument("--result_file", "--result_file",  type=str, default="../data/images/output/infer_result.txt", help="result_file")
parser.add_argument("--result_label_file", "--result_label_file",  type=str, default="../data/images/output/eval_labels.txt", help="result_label_file")
parser.add_argument("--result_top1_file", "--result_top1_file",  type=str, default="../data/images/output/eval_result_1.txt", help="result_top1_file")
parser.add_argument("--result_top5_file", "--result_top5_file",  type=str, default="../data/images/output/eval_result_5.txt", help="result_top5_file")


if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        print("please generate magicmind model first!!!")
        exit()
    model = mm.Model()
    model.deserialize_from_file(args.magicmind_model)

    name_map = load_name(args.name_file)
    record = Record(args.result_file)
    result_label = Record(args.result_label_file)
    result_top1 = Record(args.result_top1_file)
    result_top5 = Record(args.result_top5_file)
    with mm.System():
        dev = mm.Device()
        dev.id = args.device_id
        econfig = mm.Model.EngineConfig()
        econfig.device_type = "MLU"
        engine = model.create_i_engine(econfig)
        assert engine != None, "Failed to create engine"
        context = engine.create_i_context()
        queue = dev.create_queue()
        assert queue != None

        inputs = context.create_inputs()
        outputs = []

        dataset = imagenet_dataset(val_txt = args.label_file, image_file_path = args.image_dir, count = args.image_num)
        count = 0
        print("Start run ...")
        from tqdm import tqdm
        for img, label in tqdm(dataset, total=args.image_num):
            data = preprocess(img, transpose = True)
            data = np.expand_dims(data, 0)
            inputs[0].from_numpy(data)
            inputs[0].to(dev)
            
            assert context.enqueue(inputs, outputs, queue).ok()
            assert queue.sync().ok()

            index = outputs[0].asnumpy()[0].argsort()[::-1]
            record.write("top5 result:", False)
            result_label.write("[%d]: %d"%(count, int(label)), False)
            result_top1.write("[%d]: %d"%(count, index[0]), False)
            for i in range(5):
                idx = index[i]
                name = name_map[idx]
                record.write("%d [%s]"%(i, name), False)
                result_top5.write("[%d]: %d"%(count, idx), False)
            count += 1
