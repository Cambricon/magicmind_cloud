set -e
set -x
PRECISION=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3

MM_MODEL="${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}"

if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];then
    echo "$MM_MODEL not exist,please go to gen_model folder and run this command:bash run.sh $PRECISION $SHAPE_MUTABLE $BATCH_SIZE!!!"
    exit 1
fi

FILE_LIST="$DATASETS_PATH/ucfTrainTestlist/testlist01.txt" #testlist01.txt 包含3783条视频 推理时间较长 请耐心等待
THIS_OUTPUT_DIR="$PROJ_ROOT_PATH/data/output/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}"
if [ ! -d $THIS_OUTPUT_DIR ];then
    mkdir -p $THIS_OUTPUT_DIR
fi

cd $PROJ_ROOT_PATH/infer_cpp/
if [ ! -f infer ];then
    bash build.sh
fi
DEV_ID=0
./infer --magicmind_model $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL \
        --video_list $FILE_LIST \
        --device_id $DEV_ID \
        --batch_size $BATCH_SIZE \
        --output_dir $THIS_OUTPUT_DIR \
        --dataset_dir ${DATASETS_PATH} \
        --test_nums 500 \
        --name_file $DATASETS_PATH/ucfTrainTestlist/classInd.txt \
        --result_file $PROJ_ROOT_PATH/data/output/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}/infer_result.txt \
        --result_label_file $PROJ_ROOT_PATH/data/output/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}/eval_labels.txt \
        --result_top1_file $PROJ_ROOT_PATH/data/output/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}/eval_result_1.txt \
        --result_top5_file $PROJ_ROOT_PATH/data/output/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}/eval_result_5.txt
