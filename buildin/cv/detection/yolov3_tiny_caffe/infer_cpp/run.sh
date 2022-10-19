set -e
set -x
OUTPUT_DIR=$PROJ_ROOT_PATH/data/output
if [ ! -d $OUTPUT_DIR ];then
    mkdir -p $OUTPUT_DIR
fi

QUANT_MODE=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3

MM_MODEL="${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}"

THIS_OUTPUT_DIR="$PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}"
if [ ! -d $THIS_OUTPUT_DIR ];then
    mkdir -p $THIS_OUTPUT_DIR
    mkdir -p $THIS_OUTPUT_DIR/draw_imgs
    mkdir -p $THIS_OUTPUT_DIR/results
    mkdir -p $THIS_OUTPUT_DIR/json
fi

if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];then
    echo "$MM_MODEL not exist,please go to gen_model folder and run this command:bash run.sh $QUANT_MODE $SHAPE_MUTABLE $BATCH_SIZE!!!"
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
