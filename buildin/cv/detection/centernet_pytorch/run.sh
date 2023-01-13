#!/bin/bash
set -e
set -x

precision=force_float32
shape_mutable=true
batch_size=1
save_img=true
languages=infer_cpp
img_num=10


# 0. convert model
cd $PROJ_ROOT_PATH/export_model 
bash run.sh $batch_size

# 1. gen_model
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
    bash run.sh $precision $shape_mutable $save_img $img_num
fi


# ### 3.eval
OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/${languages}_output_${precision}_${shape_mutable}_${batch_size}
echo "output dir: $OUTPUT_DIR"
python $UTILS_PATH/compute_coco_mAP.py --file_list $UTILS_PATH/coco_file_list_5000.txt \
                                       --result_dir $OUTPUT_DIR \
                                       --ann_dir $DATASETS_PATH \
                                       --data_type val2017 \
                                       --json_name $OUTPUT_DIR \
                                       --image_num $img_num