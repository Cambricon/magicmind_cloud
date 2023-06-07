set -e
set -x

PRECISION=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=1

MM_MODEL=${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}

if [ ! -d $PROJ_ROOT_PATH/data/output/ ];then
    mkdir -p $PROJ_ROOT_PATH/data/output/
fi
OUTPUT_PKL=$PROJ_ROOT_PATH/data/output/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}.pkl

if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];then
    echo "$MM_MODEL not exist,please go to gen_model folder and run this command:bash run.sh $PRECISION $SHAPE_MUTABLE $BATCH_SIZE!!!"
    exit 1
fi

cd $PROJ_ROOT_PATH/infer_python/
DEV_ID=0
# test_nums eauals to -1 means all images
python infer.py $PROJ_ROOT_PATH/export_model/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
                $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL \
                --eval bbox \
                --out $OUTPUT_PKL \
                --device_id $DEV_ID

