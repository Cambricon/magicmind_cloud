#bin/bash
set -e
set -x
#example:bash run.sh force_float16 false 4
QUANT_MODE=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3

if [ ! -d $PROJ_ROOT_PATH/data/mm_model ];then
    mkdir -p $PROJ_ROOT_PATH/data/mm_model
fi

cd $PROJ_ROOT_PATH/gen_model/
if [ -f $PROJ_ROOT_PATH/data/mm_model/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} ];
then
    echo "magicmind model: $PROJ_ROOT_PATH/data/mm_model/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} already exist!"
else 
    echo "generate Magicmind model begin..."
    python gen_model.py --quant_mode ${QUANT_MODE} \
                        --batch_size $BATCH_SIZE --shape_mutable ${SHAPE_MUTABLE} \
                        --datasets_dir $DATASETS_PATH \
                        --caffe_prototxt $PROJ_ROOT_PATH/data/models/c3d_resnet18_r2_ucf101.prototxt \
                        --caffe_model $PROJ_ROOT_PATH/data/models/c3d_resnet18_ucf101_r2_ft_iter_20000.caffemodel \
                        --mm_model $PROJ_ROOT_PATH/data/mm_model/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}
    echo "Generate model done, model save to $PROJ_ROOT_PATH/data/mm_model/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}"
fi



        
