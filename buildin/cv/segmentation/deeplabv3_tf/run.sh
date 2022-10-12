#!/bin/bash
set -e
set -x
quant_mode=force_float32
shape_mutable=true
image_num=1000
input_size=513
threads=1
network=deeplabv3_tf
language=infer_cpp
### 0.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh

### 1.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh quant_mode shape_mutable
bash run.sh ${quant_mode} ${shape_mutable}

### 2 infer
cd $PROJ_ROOT_PATH/${language}
#bash run.sh quant_mode shape_mutable image_num
bash run.sh ${quant_mode} ${shape_mutable} ${image_num}

### 3.eval and perf
#bash $PROJ_ROOT_PATH/benchmark/eval.sh quant_mode shape_mutable image_num ways
bash $PROJ_ROOT_PATH/benchmark/eval.sh ${quant_mode} ${shape_mutable} ${image_num} ${language}
#bash $PROJ_ROOT_PATH/benchmark/perf.sh quant_mode shape_mutable threads input_size
bash $PROJ_ROOT_PATH/benchmark/perf.sh ${quant_mode} ${shape_mutable} ${threads} ${input_size}

###4. compare eval and perf result
python $MAGICMIND_CLOUD/test/compare_eval.py --metric voc_miou --output_file $PROJ_ROOT_PATH/benchmark/benchmark.csv --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${language}_output_${quant_mode}_${shape_mutable}_benchmark_ok.csv --model ${network}
python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/output/${quant_mode}_${shape_mutable}_${input_size}_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${quant_mode}_${shape_mutable}_${input_size}_log_perf --model ${network}

