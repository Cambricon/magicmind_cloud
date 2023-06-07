#bin/bash
set -e
set -x
PRECISION=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3

if [ ! -d $PROJ_ROOT_PATH/data/mm_model ];then
    mkdir -p $PROJ_ROOT_PATH/data/mm_model
fi

if [ -f $PROJ_ROOT_PATH/data/mm_model/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} ];
then
    echo "magicmind model: $PROJ_ROOT_PATH/data/mm_model/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} already exist!"
else 
    echo "generate Magicmind model begin..."
    cd $PROJ_ROOT_PATH/gen_model/
    python gen_model.py --precision ${PRECISION} \
                        --batch_size $BATCH_SIZE \
                        --shape_mutable ${SHAPE_MUTABLE} \
                        --onnx_model $PROJ_ROOT_PATH/data/models/maskrcnn.onnx \
                        --mm_model $PROJ_ROOT_PATH/data/mm_model/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                        --input_size 224
    echo "Generate model done, model save to $PROJ_ROOT_PATH/data/mm_model/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}"
fi