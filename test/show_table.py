import os
import prettytable

ROOT_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], "../"))
report_file="%s/report.csv"%ROOT_DIR

with open(report_file, "r") as f:
    tb = prettytable.from_csv(f)
tb.align = "l"
print(tb)

