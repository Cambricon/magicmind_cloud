#bin/bash
set -e
set -x
QUANT_MODE=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3

#example<<bash run.sh force_float16 false 4
mkdir -p $PROJ_ROOT_PATH/data/mm_model

if [ -f $PROJ_ROOT_PATH/data/mm_model/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} ];
then
    echo "magicmind model: $PROJ_ROOT_PATH/data/mm_model/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} already exist!"
else 
    echo "generate Magicmind model begin..."
    cd $PROJ_ROOT_PATH/gen_model/
    python gen_model.py --quant_mode ${QUANT_MODE} \
                        --batch_size $BATCH_SIZE \
                        --shape_mutable ${SHAPE_MUTABLE} \
                        --caffe_prototxt  $PROJ_ROOT_PATH/data/models/yolov3_tiny.prototxt \
                        --caffe_model $PROJ_ROOT_PATH/data/models/yolov3_tiny.caffemodel \
                        --datasets_dir $DATASETS_PATH/val2017 \
                        --mm_model $PROJ_ROOT_PATH/data/mm_model/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}
    echo "Generate model done, model save to $PROJ_ROOT_PATH/data/mm_model/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}"
fi



        
