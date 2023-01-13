set -e
set -x

PRECISION=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=1

MM_MODEL=${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
THIS_OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
if [ ! -d $THIS_OUTPUT_DIR ];then
    mkdir -p $THIS_OUTPUT_DIR
    mkdir -p $THIS_OUTPUT_DIR/bbox_imgs
    mkdir -p $THIS_OUTPUT_DIR/segmask_imgs
    mkdir -p $THIS_OUTPUT_DIR/results
    mkdir -p $THIS_OUTPUT_DIR/json
fi

if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];then
    echo "$MM_MODEL not exist,please go to gen_model folder and run this command:bash run.sh $PRECISION $SHAPE_MUTABLE $BATCH_SIZE!!!"
    exit 1
fi

cd $PROJ_ROOT_PATH/infer_python/
DEV_ID=0
# test_nums eauals to -1 means all images
python infer.py --magicmind_model $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL \
                --device_id $DEV_ID \
                --image_dir ${DATASETS_PATH}/val2017 \
                --label_dir ${UTILS_PATH}/coco.names \
                --output_img_dir $THIS_OUTPUT_DIR/bbox_imgs \
                --output_maskimg_dir $THIS_OUTPUT_DIR/segmask_imgs \
                --output_pred_dir $THIS_OUTPUT_DIR/results \
                --save_imgname_dir $THIS_OUTPUT_DIR/json \
                --save_img 0 \
                --score_th 0.3 \
                --save_mask 0 \
                --input_size 800 \
                --test_nums -1