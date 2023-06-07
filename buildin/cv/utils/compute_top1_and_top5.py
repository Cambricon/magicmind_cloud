import numpy as np
import argparse
from utils import Record
def get_args():
    parser = argparse.ArgumentParser(description='Calculate the TOP1/5 of imagenet dataset')

    parser.add_argument("--result_label_file", dest = 'result_label_file', help = "result label txt",
                        default = "/mm_ws/proj/modelzoo/magicmind_cloud/buildin/cv/classification/resnet50_onnx/data/images/output/eval_labels.txt", type = str)
        
    parser.add_argument("--result_1_file", dest = 'result_1_file', help = "top1 accuracy result file path",
                        default = "/mm_ws/proj/modelzoo/magicmind_cloud/buildin/cv/classification/resnet50_onnx/data/images/output/eval_result_1.txt", type = str)

    parser.add_argument("--result_5_file", dest = 'result_5_file', help = "top5 accuracy result file path",
                        default = "/mm_ws/proj/modelzoo/magicmind_cloud/buildin/cv/classification/resnet50_onnx/data/images/output/eval_result_5.txt", type = str)
    parser.add_argument("--top1andtop5_file", dest = 'top1andtop5_file', help = "top1 and top5 accuracy", default = "/mm_ws/proj/modelzoo/magicmind_cloud/buildin/cv/classification/resnet50_onnx/data/images/output/eval_result.txt")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
        
    top1_count = 0
    top5_count = 0
    total_count = 0
    
    with open(args.result_1_file, "r") as r1:
        top1_results = r1.readlines()
    with open(args.result_5_file, "r") as r5:
        top5_results = r5.readlines()
    with open(args.result_label_file, "r") as l:
        labels= l.readlines()

    for i, label in enumerate(labels):
        if label == top1_results[i]:
            top1_count += 1
        for top5_result in top5_results[i*5:(i+1)*5]:
            if label == top5_result:
                top5_count += 1
        total_count += 1
       
    top1 = float(top1_count) / float(total_count)
    top5 = float(top5_count) / float(total_count)
    print("top1: ", top1)
    print("top5: ", top5)
    result_top1andtop5 = Record(args.top1andtop5_file)
    result_top1andtop5.write("top1 accuracy: %f"%(top1), False)
    result_top1andtop5.write("top5 accuracy: %f"%(top5), False)
