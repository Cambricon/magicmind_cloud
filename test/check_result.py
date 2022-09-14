import os
import prettytable
import csv
import sys

if len(sys.argv) == 3:
    benchmark_csv = sys.argv[1]
    benchmark_ok_csv = sys.argv[2]
else:
    benchmark_csv = os.path.join(os.getenv("PROJ_ROOT_PATH"), "benchmark","benchmark.csv")
    benchmark_ok_csv = os.path.join(os.getenv("PROJ_ROOT_PATH"), "benchmark","benchmark_ok.csv")

if not os.path.exists(benchmark_csv):
    raise Exception(f"{benchmark_csv} not exists")
elif not os.path.exists(benchmark_ok_csv):
    raise Exception(f"{benchmark_ok_csv} not exists")

def load_csv(csv_file):
    with open(csv_file,'r') as f:
        reader = csv.DictReader(f)
        result = [x for x in reader]
    return result

def check(value1, value2, thresh = 0.05):
    margin = abs(value1 - value2) / abs(value1)
    if margin > thresh:
        return False
    else:
        return True

status_list = list()
tb = prettytable.PrettyTable()
benchmarks = load_csv(benchmark_csv)
benchmarks_ok = load_csv(benchmark_ok_csv)
tb.field_names = [x for x in benchmarks_ok[0].keys()]

for bench, bench_ok in zip(benchmarks, benchmarks_ok):
    check_ok = True
    err_msg = list()
    acc = eval(bench["eval"])
    acc_ok = eval(bench_ok["eval"])

    if not check(acc_ok, acc):
        check_ok = False

    if check_ok:
        status_list.append("pass")
    else:
        status_list.append("fail")
    
    value = [] 
    for k in bench_ok.keys():
        if k in ["fps", "eval"]:
            v = eval(bench[k])
            v_ok = eval(bench_ok[k])
            value.append("%.3f (%.3f)"%(v,v_ok))
        else:
            value.append(bench[k])
    tb.add_row(value)

tb.add_column("status",status_list)
tb.align = "c"
tb.max_width = 40
print(tb)

if "fail" in  status_list:
    exit(1)

