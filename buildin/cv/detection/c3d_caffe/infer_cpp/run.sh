set -e
set -x
cd $PROJ_ROOT_PATH/export_model/
bash run.sh

cd $PROJ_ROOT_PATH/infer_cpp/
bash build.sh

QUANT_MODE=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3

MM_MODEL="${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}"

if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];then
    cd $PROJ_ROOT_PATH/gen_model/
    bash run.sh $QUANT_MODE $SHAPE_MUTABLE $BATCH_SIZE
fi
FILE_LIST="$DATASETS_PATH/ucfTrainTestlist/testlist01.txt" #testlist01.txt 包含3783条视频 推理时间较长 请耐心等待
THIS_OUTPUT_DIR="$PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}"
mkdir -p $THIS_OUTPUT_DIR

cd $PROJ_ROOT_PATH/infer_cpp/
DEV_ID=0
./infer --magicmind_model $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL \
        --video_list $FILE_LIST \
        --device_id $DEV_ID \
        --batch_size $BATCH_SIZE \
        --output_dir $THIS_OUTPUT_DIR \
        --dataset_dir ${DATASETS_PATH} \
        --name_file $DATASETS_PATH/ucfTrainTestlist/classInd.txt \
        --result_file $PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}/infer_result.txt \
        --result_label_file $PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}/eval_labels.txt \
        --result_top1_file $PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}/eval_result_1.txt \
        --result_top5_file $PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}/eval_result_5.txt
