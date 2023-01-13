#!/bin/bash
set -e
set -x
precision=force_float32
batch_size=1
image_num=1000
languages=infer_cpp
### 0.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh

### 1.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
bash run.sh $precision $batch_size

### 2 infer
cd $PROJ_ROOT_PATH/$languages
bash run.sh $precision $image_num

### 3.eval and perf
python $UTILS_PATH/compute_voc_mIOU_eval.py --image_num $image_num \
                                            --language $languages \
                                            --pred_dir $PROJ_ROOT_PATH/data/output/${languages}_output_${precision}
