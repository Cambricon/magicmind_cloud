#!/bin/bash
set -e
set -x
QUANT_MODE=$1
SHAPE_MUTABLE=$2
BATCH=$3
IMAGE_NUM=$4
if [ -d $PROJ_ROOT_PATH/data/json ];
then 
    echo "folder $PROJ_ROOT_PATH/data/json already exist."
else
    mkdir $PROJ_ROOT_PATH/data/json
fi 
COMPUTE_COCO(){
    QUANT_MODE=$1
    SHAPE_MUTABLE=$2
    BATCH=$3
    IMG_NUM=$4
    python $UTILS_PATH/compute_coco_mAP.py --file_list $DATASETS_PATH/file_list_5000.txt \
                                           --result_dir $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH} \
                                           --ann_dir $DATASETS_PATH \
                                           --data_type val2017 \
                                           --json_name $PROJ_ROOT_PATH/data/json/centernet_pytorch_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH} \
                                           --image_num ${IMG_NUM} 2>&1 |tee $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/log_eval
}

#dynamic
if [ $# != 0 ];
then 
    COMPUTE_COCO ${QUANT_MODE} ${SHAPE_MUTABLE} ${BATCH} ${IMAGE_NUM}
else  
    echo "Parm doesn't exist, run benchmark"
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh 1
    for quant_mode in force_float32 force_float16 qint8_mixed_float16
    do
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $quant_mode true 1
        for batch in 1
        do
            cd $PROJ_ROOT_PATH/infer_cpp
            bash run.sh $quant_mode true $batch 1000
            COMPUTE_COCO $quant_mode true $batch 1000
            python $MAGICMIND_CLOUD/test/compare_eval.py --metric cocomAP --output_file $PROJ_ROOT_PATH/data/output/infer_cpp_output_${quant_mode}_true_${batch}/log_eval --output_ok_file $PROJ_ROOT_PATH/data/output_ok/infer_cpp_output_${quant_mode}_true_${batch}_log_eval --model centernet_pytorch
        done
    done
fi
## static
#for quant_mode in force_float32 force_float16 qint8_mixed_float16
#do
#  for batch in 1
#  do
#    cd $PROJ_ROOT_PATH/export_model
#    bash run.sh $batch
#    cd $PROJ_ROOT_PATH/gen_model
#    bash run.sh $quant_mode false $batch 0.001 0.65 1000
#    cd $PROJ_ROOT_PATH/infer_python
#    bash run.sh $quant_mode false 1 100
#    COMPUTE_COCO $quant_mode false $batch 100
#  done
#done
