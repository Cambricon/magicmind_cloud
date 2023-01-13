#!/bin/bash
set -e
set -x

precision=force_float32
shape_mutable=true
batch_size=1

# 0. convert model
pip install -r requirement.txt
cd $PROJ_ROOT_PATH/export_model 
bash run.sh $batch_size

# 1. gen_model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh precision shape_mutable batch_size conf iou max_det
bash run.sh $precision $shape_mutable $batch_size 0.001 0.65 1000

# 2. build infer_cpp and infer
cd $PROJ_ROOT_PATH/infer_cpp
#bash run.sh precision shape_mutable batch_size image_num(-1: all images)
bash run.sh $precision $shape_mutable $batch_size -1

### 3.eval 
cd $PROJ_ROOT_PATH/export_model/Pytorch_Retinaface/widerface_evaluate
python3 setup.py build_ext --inplace
python3 evaluation.py -p $PROJ_ROOT_PATH/data/output/infer_cpp_output_${precision}_true_${batch_size}/pred_txts \
                      -g $PROJ_ROOT_PATH/export_model/Pytorch_Retinaface/widerface_evaluate/ground_truth