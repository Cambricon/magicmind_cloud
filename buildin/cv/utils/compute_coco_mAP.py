from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import glob
import os
import argparse
import json
import time
import sys
sys.path.append(os.path.join(os.getenv("MAGICMIND_CLOUD"), "test"))
from record_result import write_result
class_dict = { 'person':1,'bicycle':2,'car':3,
               'motorbike':4,'aeroplane':5,'bus':6,
               'train':7,'truck':8,'boat':9,
               'traffic light':10,'fire hydrant':11,'stop sign':13,
               'parking meter':14,'bench':15,'bird':16,
               'cat':17,'dog':18,'horse':19,
               'sheep':20,'cow':21,'elephant':22,
               'bear':23,'zebra':24,'giraffe':25,
               'backpack':27,'umbrella':28,'handbag':31,
               'tie':32,'suitcase':33,'frisbee':34,
               'skis':35,'snowboard':36,'sports ball':37,
               'kite':38,'baseball bat':39,'baseball glove':40,
               'skateboard':41,'surfboard':42,'tennis racket':43,
               'bottle':44,'wine glass':46,'cup':47,
               'fork':48,'knife':49,'spoon':50,
               'bowl':51,'banana':52,'apple':53,
               'sandwich':54,'orange':55,'broccoli':56,
               'carrot':57,'hot dog':58,'pizza':59,
               'donut':60,'cake':61,'chair':62,
               'sofa':63,'pottedplant':64,'bed':65,
               'diningtable':67,'toilet':70,'tvmonitor':72,
               'laptop':73,'mouse':74,'remote':75,
               'keyboard':76,'cell phone':77,'microwave':78,
               'oven':79,'toaster':80,'sink':81,
               'refrigerator':82,'book':84,'clock':85,
               'vase':86,'scissors':87,'teddy bear':88,
               'hair drier':89,'toothbrush':90 }

def get_args():
    """
    This function creates a command-line argument parser and returns parsed arguments

    :returns: parsed command-line arguments (in the form of argparse.Namespace)
    """
    parser = argparse.ArgumentParser(description='Calculate the mAP of coco dataset')

    parser.add_argument("--file_list", dest = 'file_list', help =
                        "File list that use to calculate mAP",
                        default = "./file_list", type = str)
    parser.add_argument("--result_dir", dest = "result_dir", help =
                        "The inference result of the file, gernerate json file by results",
                        default = './',type = str)
    parser.add_argument("--ann_dir", dest = "ann_dir", help =
                        "The annotation file directory",
                        default = './',type = str)
    parser.add_argument("--data_type", dest = "data_type", help =
                        "The data type. e.g. val2014, val2015, val2017",
                        default = 'val2017',type = str)
    parser.add_argument("--json_name", dest = "json_name", help =
                        "name of the output file(.json)",
                        default = 'results',type = str)
    parser.add_argument("--img_dir", dest = "img_dir", help = "coco datasets path", default = "../datasets/", type = str)
    parser.add_argument("--image_num", dest = "image_num", help = "image number", default = 10, type = int)
    parser.add_argument("--infer_mode", dest = "infer_mode", help = "infer_mode which used to infer model", default = "infer_python", type = str)
    return parser.parse_args()

def parse_output(input):
    objs = []
    f = open(input)
    for line in f.readlines():
        objs.append(line.split(','))
    f.close()
    return objs

def get_bbox(result_objs):

    bbox_data = [float(value) for value in result_objs[2:]]
    #bbox_res = [bbox_data[0] * bbox_data[4],
    #            bbox_data[1] * bbox_data[5],
    #            (bbox_data[2] - bbox_data[0]) * bbox_data[4],
    #            (bbox_data[3] - bbox_data[1]) * bbox_data[5]]
    bbox_res = [bbox_data[0] ,
                bbox_data[1] ,
                (bbox_data[2] - bbox_data[0]) ,
                (bbox_data[3] - bbox_data[1])]
    return bbox_res

def generate_json_file(file_list, result_dir, clases_list, output_file, img_dir, image_num = -1):

    img_list_full = []
    f = open(file_list)
    num = 0 
    for img in f.readlines():
        img = os.path.join(img_dir, img)
        img_list_full.append(img)
        num += 1
        if num >= image_num and image_num != -1:
            break
    f.close()

    json_file_name = '%s.json'%(output_file)
    img_ids = []

    with open(json_file_name,'w+') as fp:
        fp.write("[\n")

        for ii, img in enumerate(img_list_full):
            img_name = os.path.splitext(os.path.basename(img))[0]
            index = int(img_name.split('_')[-1].lstrip('0'))
            img_ids.append(index)
            result_objs = parse_output(result_dir + '/' + img_name + '.txt')

            for i in range(len(result_objs)):
                img_id = index
                category_id = class_dict[result_objs[i][0]]
                score = float(result_objs[i][1])
                bbox = get_bbox(result_objs[i])
                json.dump({'image_id':img_id , 'category_id':category_id, 'bbox':bbox, 'score':score}, fp)
                if ii == len(img_list_full) - 1 and i == len(result_objs) - 1:
                    fp.write("]\n")
                else:
                    fp.write(",\n")

    return img_ids, json_file_name

def update_json_meanAp(json_data, meanAp, key):
    if isinstance(json_data, dict):
        for k in json_data:
            if k == key:
               json_data[k] = meanAp
            elif isinstance(json_data[k], dict):
               update_json_meanAp(json_data[k], meanAp, key)

class redirect:
    content = ""
    def write(self, str):
        self.content += str
    def flush(self):
        self.content = ""

if __name__ == "__main__":
    args = get_args()
    img_list = args.file_list
    result_dir = args.result_dir
    json_file  = args.json_name
    img_dir = args.img_dir

    result_dir = result_dir.rstrip('/')

    img_ids, res_file = generate_json_file(img_list, result_dir, class_dict, json_file, img_dir, args.image_num)

    print('num of images: %d'%(len(img_ids)))
    time.sleep(2)

    ann_type = ['segm', 'bbox', 'keypoints']
    ann_type = ann_type[1]
    prefix = 'person_keypoints' if ann_type == 'keypoints' else 'instances'

    data_dir = args.ann_dir
    data_type = args.data_type
    ann_file = '%s/annotations/%s_%s.json'%(data_dir, prefix, data_type)
    coco_gt = COCO(ann_file)

    #res_file = 'instances_val2014_fakebbox100_results.json'
    coco_dt = coco_gt.loadRes(res_file)

    coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    write_result(**{"infer_mode": args.infer_mode, "dataset": "coco", "metric":"map", "eval": coco_eval.stats[1]})
    input_json_file = os.getenv('OUTPUT_JSON_FILE','')
    if os.path.isfile(input_json_file):
        r = redirect()
        sys.stdout = r
        coco_eval.summarize()
        file_in = open(input_json_file, "r")
        json_data = json.load(file_in)
        update_json_meanAp(json_data, r.content, 'meanAp')
        file_in.close()
        file_out = open(input_json_file, "w")
        json.dump(json_data, file_out,indent=2)
        file_out.close()
        r.flush()
