#!/bin/bash
set -e
set -x
precision=force_float32
shape_mutable=true
batch_size=1
languages=infer_cpp
### 0.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh

### 1.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
bash run.sh $precision $shape_mutable $batch_size

### 2 infer_cpp
cd $PROJ_ROOT_PATH/infer_cpp
bash run.sh $precision $shape_mutable $batch_size

### 3.eval
python $UTILS_PATH/compute_voc_mIOU_segnet.py --output_dir $PROJ_ROOT_PATH/data/output/${languages}_output_${precision}_${shape_mutable}_${batch_size}