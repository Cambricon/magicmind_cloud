#!/usr/bin/python3

import os
import csv
import argparse
import sys

root_dir = os.getenv("PROJ_ROOT_PATH")
csv_path = os.path.join(root_dir, "benchmark", "benchmark.csv")
fieldnames = ["model", "language", "dataset","metric", "eval", "fps"]

parser = argparse.ArgumentParser()
for key in fieldnames:
    parser.add_argument("--%s"%key)

def write_result(**kw):
    model = kw.get("model", str(os.environ.get("PROJ_ROOT_PATH")).split("/")[-1])
    value = {k:kw.get(k,None) for k in fieldnames[1:]}
    value = {k: v for k, v in value.items() if v is not None}

    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = [x for x in reader]
    else:
        rows = []
    for row in rows:
        if model  == row[fieldnames[0]]:
            row.update(value)
    if model not in [x[fieldnames[0]] for x in rows]:
        row = value
        row[fieldnames[0]] = model
        rows.append(row)
    with open(csv_path,'w',newline='')as c:
        writer = csv.DictWriter(c, fieldnames=fieldnames)
        if len(rows) == 1:
            writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    param = parser.parse_args()
    dic = vars(param)
    dic = {k: v for k, v in dic.items() if v is not None}
    write_result(**dic)

