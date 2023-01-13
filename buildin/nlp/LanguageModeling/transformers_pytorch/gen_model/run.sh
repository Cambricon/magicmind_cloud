#bin/bash
set -e
set -x
PRECISION=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3

#example<<bash run.sh force_float16 false 4
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
                        --pt_model $PROJ_ROOT_PATH/data/models/traced.pt  \
                        --mm_model $PROJ_ROOT_PATH/data/mm_model/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
    echo "Generate model done, model save to $PROJ_ROOT_PATH/data/mm_model/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}"
fi



        
