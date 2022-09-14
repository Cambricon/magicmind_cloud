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
      #compute_top1_and_top5
      python $UTILS_PATH/compute_top1_and_top5.py --result_label_file $PROJ_ROOT_PATH/data/output/${quant_mode}_${shape_mutable}_${batch}/eval_labels.txt \
                                                  --result_1_file $PROJ_ROOT_PATH/data/output/${quant_mode}_${shape_mutable}_${batch}/eval_result_1.txt \
                                                  --result_5_file $PROJ_ROOT_PATH/data/output/${quant_mode}_${shape_mutable}_${batch}/eval_result_5.txt \
                                                  --top1andtop5_file $PROJ_ROOT_PATH/data/output/${quant_mode}_${shape_mutable}_${batch}/eval_result.txt
      #compare
      python $MAGICMIND_CLOUD/test/compare_eval.py  --metric top1andtop5 \
                                                    --output_file $PROJ_ROOT_PATH/data/output/${quant_mode}_${shape_mutable}_${batch}/eval_result.txt \
                                                    --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${quant_mode}_${shape_mutable}_${batch}_eval_result.txt \
                                                    --model c3d_caffe
    done
  done
done
