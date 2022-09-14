import os
import prettytable
import sys
import csv
from glob import glob
import shutil

bench_csv = "benchmark.csv"
bench_ok_csv = "benchmark_ok.csv"

if os.path.exists(bench_csv):
    os.remove(bench_csv)
    
if os.path.exists(bench_ok_csv):
    os.remove(bench_ok_csv)

all_file = glob("**/benchmark.csv", recursive=True)
all_ok_file = glob("**/benchmark_ok.csv", recursive=True)

def load_csv(csv_file):
    with open(csv_file,'r') as f:
        reader = csv.DictReader(f)
        result = [x for x in reader]
    return result

bench = [load_csv(x)[0] for x in all_file]
with open(bench_csv, "w") as f:
    csv_writter = csv.DictWriter(f, bench[0].keys())
    csv_writter.writeheader()
    csv_writter.writerows(bench)

bench_ok = [load_csv(x)[0] for x in all_ok_file]
with open(bench_ok_csv, "w") as f:
    csv_writter = csv.DictWriter(f, bench_ok[0].keys())
    csv_writter.writeheader()
    csv_writter.writerows(bench_ok)

