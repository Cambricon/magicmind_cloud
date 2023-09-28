set -e
set -x
#Example:bash run.sh force_float32 false 6
OUTPUT_DIR=$PROJ_ROOT_PATH/data/output
if [ ! -d $OUTPUT_DIR ];then
    mkdir -p $OUTPUT_DIR
fi

PRECISION=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3

MM_MODEL="${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}"

if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];then
    echo "$MM_MODEL not exist,please go to gen_model folder and run this command:bash run.sh $PRECISION $SHAPE_MUTABLE $BATCH_SIZE!!!"
    exit 1
fi

cd $PROJ_ROOT_PATH/infer_python/
DEV_ID=0
python infer.py --magicmind_model $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL \
                --dev_id $DEV_ID \
                --batch_size ${BATCH_SIZE} \
                --datasets_dir $PROJ_ROOT_PATH/data/glue_data \
                --acc_result $PROJ_ROOT_PATH/data/output/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_acc_and_f1_result.txt \
                --test_nums 1000
