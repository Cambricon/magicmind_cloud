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
from utils.utils import Record


parser = argparse.ArgumentParser()
parser.add_argument("--device_id", "--device_id",  type=int, default=0, help="device_id")
parser.add_argument("--magicmind_model", "--magicmind_model", type=str, default="../data/models/clip_onnx", help="save mm model to this path")
parser.add_argument("--image_dir", "--image_dir",  type=str, default="/nfsdata/modelzoo/datasets/cifar100/", help="cifar100 val datasets")
parser.add_argument("--result_file", "--result_file",  type=str, default="../data/images/output/infer_result.txt", help="result_file")
parser.add_argument("--result_label_file", "--result_label_file",  type=str, default="../data/images/output/eval_labels.txt", help="result_label_file")
parser.add_argument("--result_top1_file", "--result_top1_file",  type=str, default="../data/images/output/eval_result_1.txt", help="result_top1_file")
parser.add_argument("--result_top5_file", "--result_top5_file",  type=str, default="../data/images/output/eval_result_5.txt", help="result_top5_file")
parser.add_argument("--batch_size", "--batch_size",  type=int, default=1, help="batch_size")

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        print("please generate magicmind model first!!!")
        exit()
    model = mm.Model()
    model.deserialize_from_file(args.magicmind_model)

    record = Record(args.result_file)
    result_label = Record(args.result_label_file)
    result_top1 = Record(args.result_top1_file)
    result_top5 = Record(args.result_top5_file)
    clipmodel, preprocess = clip.load('ViT-B/32', "cpu")
    root = os.path.expanduser(args.image_dir)
    test = CIFAR100(root, download=True, train=False, transform=preprocess)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test.classes]).to("cpu")
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
        count = 0
        for images, labels in tqdm(DataLoader(test, batch_size=args.batch_size,drop_last=True),ncols = 80):
            inputs[0].from_numpy(images.numpy())
            inputs[0].to(dev)
            inputs[1].from_numpy(text_inputs.numpy())
            inputs[1].to(dev)
            
            assert context.enqueue(inputs, outputs, queue).ok()
            assert queue.sync().ok()
            
            logits_per_image = torch.from_numpy(outputs[0].asnumpy())
            logits_per_text = torch.from_numpy(outputs[1].asnumpy())
            probs = logits_per_image.softmax(dim=-1).cpu()
            vals, topk1 = torch.topk(probs,1)
            vals, topk5 = torch.topk(probs,5)
            for i in range(args.batch_size):
              record.write("top5 result:", False)
              result_label.write("[%d]: %d"%(count, labels[i]), False)
              result_top1.write("[%d]: %d"%(count, topk1[i][0]), False)
              for j in range(5):
                idx = topk5[i][j]
                name = test.classes[idx]
                record.write("%d [%s]"%(j, name), False)
                result_top5.write("[%d]: %d"%(count, idx), False)
              count += 1
