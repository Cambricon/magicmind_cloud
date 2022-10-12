#!/bin/bash

QUANT_MODE=$1 #force_float32/force_float16/qint8_mixed_float16
BATCH_SIZE=$2
if [ ! -f $PROJ_ROOT_PATH/data/models/pose_body25_${QUANT_MODE}_${BATCH_SIZE} ];
then
    # BODY_25
    python $PROJ_ROOT_PATH/gen_model/gen_model.py --prototxt $MODEL_PATH/pose_deploy.prototxt \
	                                          --caffemodel $MODEL_PATH/pose_iter_584000.caffemodel \
                                                  --output_model $MODEL_PATH/pose_body25_${QUANT_MODE}_${BATCH_SIZE} \
						  --batchsize $BATCH_SIZE \
						  --quant_mode $QUANT_MODE \
                                                  --calibrate_list $PROJ_ROOT_PATH/gen_model/calibrate_list.txt
fi
if [ ! -f $PROJ_ROOT_PATH/data/models/pose_coco_${QUANT_MODE}_${BATCH_SIZE} ];
then
    # COCO
    python $PROJ_ROOT_PATH/gen_model/gen_model.py --prototxt $MODEL_PATH/pose_deploy_linevec.prototxt \
	                                          --caffemodel $MODEL_PATH/pose_iter_440000.caffemodel \
                                                  --output_model $MODEL_PATH/pose_coco_${QUANT_MODE}_${BATCH_SIZE} \
						  --batchsize $BATCH_SIZE \
						  --quant_mode $QUANT_MODE \
                                                  --calibrate_list $PROJ_ROOT_PATH/gen_model/calibrate_list.txt
fi
