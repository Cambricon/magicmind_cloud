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

if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$PROJ_ROOT_PATH/data/mm_model/clip_onnx_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
else
    MAGICMIND_MODEL=$PROJ_ROOT_PATH/data/mm_model/clip_onnx_model_${PRECISION}_${SHAPE_MUTABLE}
fi

if [ -f $MAGICMIND_MODEL ];
then
    echo "magicmind model: $MAGICMIND_MODEL already exist!"
else 
    echo "generate Magicmind model begin..."
    cd $PROJ_ROOT_PATH/gen_model/
    python gen_model.py --precision ${PRECISION}  --batch_size $BATCH_SIZE --shape_mutable ${SHAPE_MUTABLE} \
                                                    --onnx_model $PROJ_ROOT_PATH/data/models/clip.onnx \
                                                    --datasets_dir $DATASETS_PATH/cifar-100-python \
                                                    --mm_model $MAGICMIND_MODEL
    echo "Generate model done, model save to $MAGICMIND_MODEL"
fi



        
