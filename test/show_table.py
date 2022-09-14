import os
import prettytable

ROOT_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], "../test/"))
image_dir_list = os.listdir(ROOT_DIR)
report_files = []
for image_dir in image_dir_list:
    if os.path.splitext(image_dir)[1] == ".csv":
        report_file = os.path.join(ROOT_DIR, image_dir)
        report_files.append(report_file)

for report_file in report_files:
    with open(report_file, "r") as f:
        tb = prettytable.from_csv(f)
    tb.align = "l"
    print(report_file)
    print(tb)


