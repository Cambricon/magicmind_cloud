set -e
set -x
#Example:bash run.sh force_float32 false 6
cd $PROJ_ROOT_PATH/export_model/
bash run.sh

cd $PROJ_ROOT_PATH/infer_cpp/
bash build.sh

OUTPUT_DIR=$PROJ_ROOT_PATH/data/output
mkdir -p $OUTPUT_DIR

QUANT_MODE=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3

MM_MODEL="${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}"

THIS_OUTPUT_DIR="$PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}"
mkdir -p $THIS_OUTPUT_DIR/draw_imgs
mkdir -p $THIS_OUTPUT_DIR/results
mkdir -p $THIS_OUTPUT_DIR/json

if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];then
    cd $PROJ_ROOT_PATH/gen_model/
    bash run.sh $QUANT_MODE $SHAPE_MUTABLE $BATCH_SIZE
fi

cd $PROJ_ROOT_PATH/infer_cpp/
DEV_ID=0
./infer --magicmind_model $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL \
        --device_id $DEV_ID \
        --batch_size ${BATCH_SIZE} \
        --image_dir ${DATASETS_PATH}/val2017 \
        --label_path ${DATASETS_PATH}/coco.names \
        --output_img_dir $THIS_OUTPUT_DIR/draw_imgs \
        --output_pred_dir $THIS_OUTPUT_DIR/results \
        --save_imgname_dir $THIS_OUTPUT_DIR/json \
        --save_img 0 \
        --save_pred 1
