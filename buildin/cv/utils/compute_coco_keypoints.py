import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils import Record


def main():
    args = argparse.ArgumentParser(description='Evaluate')
    args.add_argument('--ann_file', dest = 'ann_file', required = True, type = str, help = 'annotation file path')
    args.add_argument('--res_file', dest = 'res_file', required = True, type = str, help = 'result file path')
    args.add_argument('--res2_file', dest = 'res2_file', required = False, type = str, help = 'result file path')
    args.add_argument('--output_file', dest = 'output_file', required = False, type = str, help = 'save result to file')
    args.add_argument('--iou_type', dest = 'iou_type', required = False, default = "keypoints", type = str, help = 'save result to file')
    args = args.parse_args()

    coco_gt = COCO(args.ann_file)
    coco_dt = coco_gt.loadRes(args.res_file)
    iou_type = args.iou_type
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.imgIds = coco_gt.getImgIds()
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stat1 = coco_eval.stats[1]
    stat2 = 0.0;
    if args.res2_file is not None:
        coco_gt = COCO(args.ann_file)
        coco_dt = coco_gt.loadRes(args.res2_file)
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.params.imgIds = coco_gt.getImgIds()
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stat2 = coco_eval.stats[1]
        
    if args.output_file is not None:
        output_file = Record(args.output_file)
        output_file.write("AP IoU=0.50:%f"%(stat1), False);
        if args.res2_file is not None:
            output_file.write("AP IoU=0.50:%f"%(stat2), False);

if __name__ == "__main__":
    main()

