#!/bin/bash
set -e
set -x

precision=force_float32
shape_mutable=true
batch_size=1
save_img=true
languages=infer_python
### 0.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh

### 1.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
bash run.sh $precision $shape_mutable $batch_size

### 2.1 infer_python
if [ $languages == "infer_python" ];
then
    cd $PROJ_ROOT_PATH/infer_python
    bash run.sh $precision $shape_mutable $save_img
fi

### 2.2 infer_cpp
if [ $languages == "infer_cpp" ];
then
    cd $PROJ_ROOT_PATH/infer_cpp
    bash run.sh $precision $shape_mutable $save_img
fi

### 3.eval
python $UTILS_PATH/compute_voc_mAP.py --path $PROJ_ROOT_PATH/data/output/${languages}_output_${precision}_${shape_mutable}_1/voc_preds/ \
                                      --devkit_path $VOC2007_DATASETS_PATH/VOCdevkit 2>&1 |tee $PROJ_ROOT_PATH/data/output/${languages}_output_${precision}_${shape_mutable}_1/log_eval
