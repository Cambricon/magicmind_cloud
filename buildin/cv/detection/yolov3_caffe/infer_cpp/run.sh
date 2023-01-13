set -e
set -x
#Example:bash run.sh force_float32 false 6

PRECISION=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3

if [ $SHAPE_MUTABLE == "true" ];then
    MM_MODEL=${PRECISION}_${SHAPE_MUTABLE}_1
else
    MM_MODEL=${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
fi 

THIS_OUTPUT_DIR="$PROJ_ROOT_PATH/data/output/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}"
if [ ! -d $THIS_OUTPUT_DIR ];then
    mkdir -p $THIS_OUTPUT_DIR
    mkdir -p $THIS_OUTPUT_DIR/draw_imgs
    mkdir -p $THIS_OUTPUT_DIR/results
    mkdir -p $THIS_OUTPUT_DIR/json
fi

if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];then
    echo "$MM_MODEL not exist,please go to gen_model folder and run this command:bash run.sh $PRECISION $SHAPE_MUTABLE $BATCH_SIZE!!!"
    exit 1
fi

cd $PROJ_ROOT_PATH/infer_cpp/
if [ ! -f infer ];then
    bash build.sh
fi
DEV_ID=0
./infer --magicmind_model $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL \
        --device_id $DEV_ID \
        --batch_size ${BATCH_SIZE} \
        --image_dir ${DATASETS_PATH}/val2017 \
        --label_path ${UTILS_PATH}/coco.names \
        --output_img_dir $THIS_OUTPUT_DIR/draw_imgs \
        --output_pred_dir $THIS_OUTPUT_DIR/results \
        --save_imgname_dir $THIS_OUTPUT_DIR/json \
        --save_img 0 \
        --save_pred 1 \
        --test_nums 1000
