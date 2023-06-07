#!/bin/bash
set -e
set -x
echo "Start!"

source env.sh
# 将pytorch模型转换成onnx模型
cd $PROJ_ROOT_PATH/export_model
bash run.sh

# 编译magicmind模型
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh <precision>
bash run.sh force_float32

# 测试精度
cd $PROJ_ROOT_PATH/infer_python
bash run.sh force_float32

