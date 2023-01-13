import json
import os
import numpy as np
import cv2
import argparse
import prettytable
ROOT_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], "../"))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--output_ok_file", type=str)
parser.add_argument("--metric", type=str, choices = ["top1", "top1andtop5", "cocomAP", "vocmAP", "unet", "squad", "voc_miou", "1e5and1e4", "mrpc", "u2net", "cocoKeyPoints", "total_text"])

ROOT_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], ""))
report_file = "%s/eval_report.csv"%ROOT_DIR
# top1 compare
def compare_top1(file1, file2):
    with open(file1, "r") as f:
        lines1 = f.readlines()
    with open(file2, "r") as f:
        lines2 = f.readlines()
    old_top1 = []
    new_top1 = []
    for l1 in lines1:
        new_top1.append(float(l1.split(":")[1].strip()))
    for l2 in lines2:
        old_top1.append(float(l2.split(":")[1].strip()))
    status = new_top1[0] >= old_top1[0]
    top1_diff = new_top1[0] - old_top1[0]
    return status, top1_diff

# top1andtop5 compare
def compare_top1andtop5(file1, file2):
    with open(file1, "r") as f:
        lines1 = f.readlines()
    with open(file2, "r") as f:
        lines2 = f.readlines()
    old_top1andtop5 = []
    new_top1andtop5 = []
    for l1 in lines1:
        new_top1andtop5.append(float(l1.split(":")[1].strip()))
    for l2 in lines2:
        old_top1andtop5.append(float(l2.split(":")[1].strip()))
    status = new_top1andtop5[0] >= old_top1andtop5[0] and new_top1andtop5[1] >= old_top1andtop5[1]
    top1_diff = new_top1andtop5[0] - old_top1andtop5[0]
    top5_diff = new_top1andtop5[1] - old_top1andtop5[1]
    return status, top1_diff, top5_diff

# coco mAP
def compute_cocomAP(file1, file2):
    with open(file1, "r") as f:
        lines1 = f.readlines()
    with open(file2, "r") as f:
        lines2 = f.readlines()
    for line1 in lines1:
        if "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]" in line1:
            new_mAP_0_5to0_95 = float(line1.split("=")[-1].strip())
        if "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]" in line1:
            new_mAP_0_5 = float(line1.split("=")[-1].strip())
    for line2 in lines2:
        if "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]" in line2:
            old_mAP_0_5to0_95 = float(line2.split("=")[-1].strip())
        if "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]" in line2:
            old_mAP_0_5 = float(line2.split("=")[-1].strip())
    status = new_mAP_0_5to0_95 >= old_mAP_0_5to0_95 and new_mAP_0_5 >= old_mAP_0_5
    mAP_0_5to0_95_diff = new_mAP_0_5to0_95 - old_mAP_0_5to0_95
    mAP_0_5_diff = new_mAP_0_5 - old_mAP_0_5
    return status, mAP_0_5to0_95_diff, mAP_0_5_diff

def compute_vocmAP(file1, file2):
    with open(file1, "r") as f:
        lines1 = f.readlines()
    with open(file2, "r") as f:
        lines2 = f.readlines()
    for line1 in lines1:
        if "Mean AP" in line1:
            new_mAP = float(line1.split("=")[-1].strip())
    for line2 in lines2:
        if "Mean AP" in line2:
            old_mAP = float(line2.split("=")[-1].strip())
    status = new_mAP >= old_mAP
    mAP_diff = new_mAP - old_mAP
    return status, mAP_diff

def compute_unet(file1, file2):
    f1_dict=json.load(open(file1, "r"))
    f2_dict=json.load(open(file2, "r"))
    new_accuracy = f1_dict["results"]["mean"]["1"]["Accuracy"]
    new_dice = f1_dict["results"]["mean"]["1"]["Dice"]
    new_precision = f1_dict["results"]["mean"]["1"]["Precision"]
    new_recall = f1_dict["results"]["mean"]["1"]["Recall"]
    old_accuracy = f2_dict["results"]["mean"]["1"]["Accuracy"]
    old_dice = f2_dict["results"]["mean"]["1"]["Dice"] 
    old_precision = f2_dict["results"]["mean"]["1"]["Precision"]
    old_recall = f2_dict["results"]["mean"]["1"]["Recall"]
    status = new_accuracy >= old_accuracy
    accuracy_diff = new_accuracy - old_accuracy
    dice_diff = new_dice - old_dice
    precision_diff = new_precision - old_precision
    recall_diff = new_recall - old_recall
    return status, accuracy_diff, dice_diff, precision_diff, recall_diff

def compute_squad(file1, file2):
    with open(file1, "r") as f:
        lines1 = f.readlines()
    with open(file2, "r") as f:
        lines2 = f.readlines()
    for l1 in lines1:
        new_exact = float(l1.split(":")[-1].split(",")[1].strip(")"))
        new_f1 = float(l1.split(":")[-1].split(",")[3].strip(")"))
    for l2 in lines2:
        old_exact = float(l2.split(":")[-1].split(",")[1].strip(")"))
        old_f1 = float(l2.split(":")[-1].split(",")[3].strip(")"))
    status = new_exact >= old_exact and new_f1 >= old_f1
    exact_diff = new_exact - old_exact
    f1_diff = new_f1 - old_f1
    return status, exact_diff, f1_diff

def compute_mrpc(file1, file2):
    with open(file1, "r") as f:
        lines1 = f.readlines()
    with open(file2, "r") as f:
        lines2 = f.readlines()
        
    new_acc = float(lines1[0].split(":")[-1])
    new_f1 =  float(lines1[1].split(":")[-1])
    old_acc = float(lines2[0].split(":")[-1])
    old_f1 =  float(lines2[1].split(":")[-1])
    
    status = new_acc >= old_acc and new_f1 >= old_f1
    acc_diff = new_acc - old_acc
    f1_diff = new_f1 - old_f1
    return status, acc_diff, f1_diff
    
def compute_voc_miou(file1, file2):
    with open(file1, "r") as f:
        lines1 = f.readlines()
    with open(file2, "r") as f:
        lines2 = f.readlines()
    for l1 in lines1:
        if "miou" in l1:
            new_voc_miou = float(l1.split(",")[4])
    for l2 in lines2:
        if "miou" in l2:
            old_voc_miou = float(l2.split(",")[4])
    voc_miou_diff = new_voc_miou - old_voc_miou
    status = new_voc_miou >= old_voc_miou
    return status, voc_miou_diff

def compute_total_text(file1, file2):
    with open(file1, "r") as f:
        lines1 = f.readlines()
    with open(file2, "r") as f:
        lines2 = f.readlines()
    new_precision = float(lines1[0].split(" ")[2])
    new_recall = float(lines1[1].split(" ")[2])
    new_fmeasure = float(lines1[2].split(" ")[2])
    old_precision = float(lines2[0].split(" ")[2])
    old_recall =  float(lines2[1].split(" ")[2])
    old_fmeasure = float(lines2[2].split(" ")[2])


    status = new_precision >= old_precision
    precision_diff = new_precision - old_precision
    return status, precision_diff

def summary_top1(testcase_name, file, metric, result, top1_diff):
    header = ["testcase", "file", "metric", "status", "top1_diff"]
    show_tb = prettytable.PrettyTable()
    show_tb.align = "l"
    show_tb.field_names = header
    report_file = "%s/eval_top1_report.csv"%ROOT_DIR
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            tb = prettytable.from_csv(f)
            tb.align = "l"
    else:
        tb = prettytable.PrettyTable()
        tb.field_names = header
        tb.align = "l"

    if result:
        status = "pass"
    else:
        status = "fail"

    show_tb.add_row ([testcase_name, file.split("/")[-1], metric, status, top1_diff])
    tb.add_row ([testcase_name ,file.split("/")[-1], metric, status, top1_diff])

    print(show_tb)
    with open(report_file, "w") as f:
        f.write(tb.get_csv_string())

def summary_top1andtop5(testcase_name, file, metric, result, top1_diff, top5_diff):
    header = ["testcase", "file", "metric", "status", "top1_diff", "top5_diff"]
    show_tb = prettytable.PrettyTable()
    show_tb.align = "l"
    show_tb.field_names = header
    report_file = "%s/eval_top1andtop5_report.csv"%ROOT_DIR
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            tb = prettytable.from_csv(f)
            tb.align = "l"
    else:
        tb = prettytable.PrettyTable()
        tb.field_names = header
        tb.align = "l"

    if result:
        status = "pass"
    else:
        status = "fail"
    
    show_tb.add_row ([testcase_name, file.split("/")[-2], metric, status, top1_diff, top5_diff])
    tb.add_row ([testcase_name ,file.split("/")[-2], metric, status, top1_diff, top5_diff])
    
    print(show_tb)
    with open(report_file, "w") as f:
        f.write(tb.get_csv_string())
def summary_1e5and1e4(testcase_name, file, metric, result, res_1e5_diff, res_1e4_diff):
    header = ["testcase", "file", "metric", "status", "1e5_diff", "1e4_diff"]
    show_tb = prettytable.PrettyTable()
    show_tb.align = "l"
    show_tb.field_names = header
    report_file = "%s/eval_1e5and1e4_report.csv"%ROOT_DIR

    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            tb = prettytable.from_csv(f)
            tb.align = "l"
    else:
        tb = prettytable.PrettyTable()
        tb.field_names = header
        tb.align = "l"

    if result:
        status = "pass"
    else:
        status = "fail"
    
    show_tb.add_row ([testcase_name, file.split("/")[-2], metric, status, res_1e5_diff, res_1e4_diff])
    tb.add_row ([testcase_name ,file.split("/")[-2], metric, status, res_1e5_diff, res_1e4_diff])
   
    print(show_tb)
    with open(report_file, "w") as f:
        f.write(tb.get_csv_string())

def summary_u2net(testcase_name, file, metric, result, mae, fmeasure):
    header = ["testcase", "file", "metric", "status", "average_mae_diff", "max_fmeasure_diff"]
    show_tb = prettytable.PrettyTable()
    show_tb.align = "l"
    show_tb.field_names = header
    report_file = "%s/eval_u2net_report.csv"%ROOT_DIR
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            tb = prettytable.from_csv(f)
            tb.align = "l"
    else:
        tb = prettytable.PrettyTable()
        tb.field_names = header
        tb.align = "l"

    if result:
        status = "pass"
    else:
        status = "fail"
    
    show_tb.add_row ([testcase_name, file.split("/")[-1], metric, status, mae, fmeasure])
    tb.add_row ([testcase_name ,file.split("/")[-2], metric, status, mae, fmeasure])
 
    print(show_tb)
    with open(report_file, "w") as f:
        f.write(tb.get_csv_string())

def summary_cocoKeyPoints(testcase_name, file, metric, result, body25_diff, coco_diff):
    header = ["testcase", "file", "metric", "status", "body25_diff", "coco_diff"]
    show_tb = prettytable.PrettyTable()
    show_tb.align = "l"
    show_tb.field_names = header
    report_file = "%s/eval_cocoKeyPoints_report.csv"%ROOT_DIR

    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            tb = prettytable.from_csv(f)
            tb.align = "l"
    else:
        tb = prettytable.PrettyTable()
        tb.field_names = header
        tb.align = "l"

    if result:
        status = "pass"
    else:
        status = "fail"
    
    show_tb.add_row ([testcase_name, file.split("/")[-1], metric, status, body25_diff, coco_diff])
    tb.add_row ([testcase_name ,file.split("/")[-2], metric, status, body25_diff, coco_diff])
    
    print(show_tb)
    with open(report_file, "w") as f:
        f.write(tb.get_csv_string())

def summary_cocomAP(testcase_name, file, metric, result, mAP_0_5to0_95_diff, mAP_0_5_diff):
    header = ["testcase", "file", "metric", "status", "mAP_0_5to0_95_diff", "mAP_0_5_diff"]
    show_tb = prettytable.PrettyTable()
    show_tb.align = "l"
    show_tb.field_names = header
    report_file = "%s/eval_cocomAP_report.csv"%ROOT_DIR
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            tb = prettytable.from_csv(f)
            tb.align = "l"
    else:
        tb = prettytable.PrettyTable()
        tb.field_names = header
        tb.align = "l"

    if result:
        status = "pass"
    else:
        status = "fail"

    show_tb.add_row ([testcase_name, file.split("/")[-2], metric, status, mAP_0_5to0_95_diff, mAP_0_5_diff])
    tb.add_row ([testcase_name ,file.split("/")[-2], metric, status, mAP_0_5to0_95_diff, mAP_0_5_diff])

    print(show_tb)
    with open(report_file, "w") as f:
        f.write(tb.get_csv_string())

def summary_vocmAP(testcase_name, file, metric, result, mAP_diff):
    header = ["testcase", "file", "metric", "status", "mAP_diff"]
    show_tb = prettytable.PrettyTable()
    show_tb.align = "l"
    show_tb.field_names = header
    report_file = "%s/eval_vocmAP_report.csv"%ROOT_DIR
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            tb = prettytable.from_csv(f)
            tb.align = "l"
    else:
        tb = prettytable.PrettyTable()
        tb.field_names = header
        tb.align = "l"

    if result:
        status = "pass"
    else:
        status = "fail"

    show_tb.add_row ([testcase_name, file.split("/")[-3], metric, status, mAP_diff])
    tb.add_row ([testcase_name ,file.split("/")[-3], metric, status, mAP_diff])

    print(show_tb)
    with open(report_file, "w") as f:
        f.write(tb.get_csv_string())

def summary_unet(testcase_name, file, metric, result, accuracy_diff, dice_diff, precision_diff, recall_diff):
    header = ["testcase", "file", "metric", "status", "accuracy_diff", "dice_diff", "precision_diff", "recall_diff"]
    show_tb = prettytable.PrettyTable()
    show_tb.align = "l"
    show_tb.field_names = header
    report_file = "%s/eval_unet_report.csv"%ROOT_DIR
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            tb = prettytable.from_csv(f)
            tb.align = "l"
    else:
        tb = prettytable.PrettyTable()
        tb.field_names = header
        tb.align = "l"

    if result:
        status = "pass"
    else:
        status = "fail"

    show_tb.add_row ([testcase_name, file.split("/")[-2], metric, status, accuracy_diff, dice_diff, precision_diff, recall_diff])
    tb.add_row ([testcase_name ,file.split("/")[-2], metric, status, accuracy_diff, dice_diff, precision_diff, recall_diff])

    print(show_tb)
    with open(report_file, "w") as f:
        f.write(tb.get_csv_string())

def summary_squad(testcase_name, file, metric, result, exact_diff, f1_diff):
    header = ["testcase", "file", "metric", "status", "exact_diff", "f1_diff"]
    show_tb = prettytable.PrettyTable()
    show_tb.align = "l"
    show_tb.field_names = header
    report_file = "%s/eval_squad_report.csv"%ROOT_DIR
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            tb = prettytable.from_csv(f)
            tb.align = "l"
    else:
        tb = prettytable.PrettyTable()
        tb.field_names = header
        tb.align = "l"

    if result:
        status = "pass"
    else:
        status = "fail"

    show_tb.add_row ([testcase_name, file.split("/")[-2], metric, status, exact_diff, f1_diff])
    tb.add_row ([testcase_name ,file.split("/")[-2], metric, status, exact_diff, f1_diff])

    print(show_tb)
    with open(report_file, "w") as f:
        f.write(tb.get_csv_string())

def summary_mrpc(testcase_name, file, metric, result, acc_diff, f1_diff):
    header = ["testcase", "file", "metric", "status", "acc_diff", "f1_diff"]
    show_tb = prettytable.PrettyTable()
    show_tb.align = "l"
    show_tb.field_names = header
    report_file = "%s/eval_squad_report.csv"%ROOT_DIR
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            tb = prettytable.from_csv(f)
            tb.align = "l"
    else:
        tb = prettytable.PrettyTable()
        tb.field_names = header
        tb.align = "l"

    if result:
        status = "pass"
    else:
        status = "fail"

    show_tb.add_row ([testcase_name, file.split("/")[-2], metric, status, acc_diff, f1_diff])
    tb.add_row ([testcase_name ,file.split("/")[-2], metric, status, acc_diff, f1_diff])

    print(show_tb)
    with open(report_file, "w") as f:
        f.write(tb.get_csv_string())

def summary_voc_miou(testcase_name, file, metric, result, voc_miou_diff):
    header = ["testcase", "file", "metric", "status", "voc_miou_diff"]
    show_tb = prettytable.PrettyTable()
    show_tb.align = "l"
    show_tb.field_names = header
    report_file = "%s/eval_voc_miou_report.csv"%ROOT_DIR
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            tb = prettytable.from_csv(f)
            tb.align = "l"
    else:
        tb = prettytable.PrettyTable()
        tb.field_names = header
        tb.align = "l"

    if result:
        status = "pass"
    else:
        status = "fail"

    show_tb.add_row ([testcase_name, file.split("/")[-1], metric, status, voc_miou_diff])
    tb.add_row ([testcase_name ,file.split("/")[-1], metric, status, voc_miou_diff])

    print(show_tb)
    with open(report_file, "w") as f:
        f.write(tb.get_csv_string())

def summary_total_text(testcase_name, file, metric, result, precision_diff):
    header = ["testcase", "file", "metric", "status", "precision_diff"]
    show_tb = prettytable.PrettyTable()
    show_tb.align = "l"
    show_tb.field_names = header
    report_file = "%s/eval_total_text_report.csv"%ROOT_DIR
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            tb = prettytable.from_csv(f)
            tb.align = "l"
    else:
        tb = prettytable.PrettyTable()
        tb.field_names = header
        tb.align = "l"

    if result:
        status = "pass"
    else:
        status = "fail"

    show_tb.add_row ([testcase_name, file.split("/")[-2], metric, status, precision_diff])
    tb.add_row ([testcase_name ,file.split("/")[-2], metric, status, precision_diff])

    print(show_tb)
    with open(report_file, "w") as f:
        f.write(tb.get_csv_string())

def summary(testcase_name, file, metric, result):
    header = [ "testcase" , "file" , "metric", "similar", "status"]
    show_tb = prettytable.PrettyTable()
    show_tb.align = "l"
    show_tb.field_names = header
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            tb = prettytable.from_csv(f)
            tb.align = "l"
    else:
        tb = prettytable.PrettyTable()
        tb.field_names = header
        tb.align = "l"

    show_tb.add_row ([ testcase_name , file, metric, result, "fail"])
    tb.add_row ([ testcase_name , file, metric, result, "fail"])

    print(show_tb)
    with open(report_file, 'w') as f:
        f.write(tb.get_csv_string())

if __name__ == "__main__":
    args = parser.parse_args()
    testcase_name = args.model    
    output_file = os.path.join(args.output_file)
    output_ok_file = os.path.join(args.output_ok_file)
    if not os.path.exists(output_file):
        result = "file: %s not exists."%output_file
        summary(testcase_name, args.output_file, args.metric, result)
    elif not os.path.exists(output_ok_file):
        result = "file: %s not exists."%output_ok_file
        summary(testcase_name, args.output_file, args.metric, result)
    else:
        if args.metric == "top1":
            result, top1_diff = compare_top1(output_file, output_ok_file)
            summary_top1(testcase_name, args.output_file, args.metric, result, top1_diff)
        elif args.metric == "top1andtop5":
            result, top1_diff, top5_diff = compare_top1andtop5(output_file, output_ok_file)
            summary_top1andtop5(testcase_name, args.output_file, args.metric, result, top1_diff, top5_diff)
        elif args.metric == "1e5and1e4":
            result, top1_diff, top5_diff = compare_top1andtop5(output_file, output_ok_file)
            summary_1e5and1e4(testcase_name, args.output_file, args.metric, result, top1_diff, top5_diff)
        elif args.metric == "cocomAP":
            result, mAP_0_5to0_95_diff, mAP_0_5_diff = compute_cocomAP(output_file, output_ok_file)
            summary_cocomAP(testcase_name, args.output_file, args.metric, result, mAP_0_5to0_95_diff, mAP_0_5_diff)
        elif args.metric == "cocoKeyPoints":
            result, body25_diff, coco_diff = compare_top1andtop5(output_file, output_ok_file)
            summary_cocoKeyPoints(testcase_name, args.output_file, args.metric, result, body25_diff, coco_diff)
        elif args.metric == "vocmAP":
            result, mAP_diff = compute_vocmAP(output_file, output_ok_file)
            summary_vocmAP(testcase_name, args.output_file, args.metric, result, mAP_diff)
        elif args.metric == "unet":
            result, accuracy_diff, dice_diff, precision_diff, recall_diff = compute_unet(output_file, output_ok_file)
            summary_unet(testcase_name, args.output_file, args.metric, result, accuracy_diff, dice_diff, precision_diff, recall_diff)
        elif args.metric == "u2net":
            result, mae, fmeasure = compare_top1andtop5(output_file, output_ok_file)
            summary_u2net(testcase_name, args.output_ok_file, args.metric, result, mae, fmeasure)
        elif args.metric == "squad":
            result, exact_diff, f1_diff = compute_squad(output_file, output_ok_file)
            summary_squad(testcase_name, args.output_file, args.metric, result, exact_diff, f1_diff)
        elif args.metric == "voc_miou":
            result, voc_miou_diff = compute_voc_miou(output_file, output_ok_file)
            summary_voc_miou(testcase_name, args.output_ok_file, args.metric, result, voc_miou_diff)
        elif args.metric == "mrpc":
            result, acc_diff, f1_diff = compute_mrpc(output_file, output_ok_file)
            summary_mrpc(testcase_name, args.output_file, args.metric, result, acc_diff, f1_diff)
        elif args.metric == "total_text":
            result, precision_diff = compute_total_text(output_file, output_ok_file)
            summary_total_text(testcase_name, args.output_file, args.metric, result, precision_diff)
        else:
            result = "metric: %s not support."%args.metric
            summary(testcase_name, args.output_file, args.metric, result)

