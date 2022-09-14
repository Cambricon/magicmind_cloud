import json
import os
import numpy as np
import cv2
import argparse
import prettytable
ROOT_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], ""))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--output_ok_file", type=str)
parser.add_argument("--metric", type=str, choices = ["normal", "tacotron"], default="normal")
report_file = "%s/perf_report.csv"%ROOT_DIR

# top1andtop5 compare
def compare_perf(file1, file2):
    with open(file1, "r") as f:
        lines1 = f.readlines()
    with open(file2, "r") as f:
        lines2 = f.readlines()
    for line1 in lines1:
        if "Throughput (qps)" in line1:
            new_qps = float(line1.split(":")[1].strip())
    for line2 in lines2:
        if "Throughput (qps)" in line2:
            old_qps = float(line2.split(":")[1].strip()) 
    result = new_qps / old_qps
    return result

def compare_perf_tacotron(file1, file2):
    with open(file1, "r") as f:
        lines1 = f.readlines()
    with open(file2, "r") as f:
        lines2 = f.readlines()
    for line1 in lines1:
        if "tacotron2_items_per_sec average" in line1:
            new_qps = float(line1.split("=")[1].strip())
    for line2 in lines2:
        if "tacotron2_items_per_sec average" in line2:
            old_qps = float(line2.split("=")[1].strip())
    result = new_qps / old_qps
    return result

def summary(testcase_name, file, result):
    header = ["testcase", "file", "status", "similar"]
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

    if isinstance(result, float):
        status = "pass" 
        if result >= 0.95 and result <= 1.05: 
            status = "pass"
        elif result > 1.05:
            status = "option"
        else:
            status = "fail"
    else:
        status = "fail"
    
    show_tb.add_row ([testcase_name, file.split("/")[-1], status, result])
    tb.add_row ([testcase_name, file.split("/")[-1], status, result])
    
    print(show_tb)
    with open(report_file, "w") as f:
        f.write(tb.get_csv_string())

if __name__ == "__main__":
    args = parser.parse_args()
    testcase_name = args.model    
    output_file = os.path.join(args.output_file)
    output_ok_file = os.path.join(args.output_ok_file)
    if not os.path.exists(output_file):
        result = "file: %s not exists."%output_file
    elif not os.path.exists(output_ok_file):
        result = "file: %s not exists."%output_ok_file
    elif args.metric == "tacotron":
        result = compare_perf_tacotron(output_file, output_ok_file)
    else:
        result = compare_perf(output_file, output_ok_file)
    summary(testcase_name, args.output_file, result)
