#bin/bash
set -e 
set -x

cd $PROJ_ROOT_PATH/export_model/
bash run.sh
#static cpp
DEV_ID=0
cd $PROJ_ROOT_PATH/infer_cpp/
bash build.sh
for quant_mode in force_float32 force_float16 qint8_mixed_float16
do
  for shape_mutable in false
  do
    for batch in 1 4 8
    do
      MM_MODEL="${quant_mode}_${shape_mutable}_${batch}"
      if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];then
          cd $PROJ_ROOT_PATH/gen_model/
          bash run.sh $quant_mode $shape_mutable $batch
      fi
      #infer cpp
      cd $PROJ_ROOT_PATH/infer_cpp/
      bash run.sh $quant_mode $shape_mutable $batch
      #compute coco
      THIS_OUTPUT_DIR="$PROJ_ROOT_PATH/data/output/${quant_mode}_${shape_mutable}_${batch}"
      python $UTILS_PATH/compute_coco_mAP.py  --file_list ${THIS_OUTPUT_DIR}/json/image_name.txt \
                                              --result_dir $THIS_OUTPUT_DIR/results \
                                              --ann_dir $DATASETS_PATH/ \
                                              --data_type 'val2017' \
                                              --json_name $THIS_OUTPUT_DIR/json/${quant_mode}_${shape_mutable}_${batch} \
                                              --img_dir $DATASETS_PATH/val2017 \
                                              --image_num 5000 2>&1 | tee $PROJ_ROOT_PATH/data/output/${quant_mode}_${shape_mutable}_${batch}_log_eval
      #compare_eval
      mkdir -p $PROJ_ROOT_PATH/data/output_ok/
      python $MAGICMIND_CLOUD/test/compare_eval.py  --metric cocomAP \
                                                    --output_file $PROJ_ROOT_PATH/data/output/${quant_mode}_${shape_mutable}_${batch}_log_eval \
                                                    --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${quant_mode}_${shape_mutable}_${batch}_log_eval \
                                                    --model yolov3_tiny_caffe
    done
  done
done
