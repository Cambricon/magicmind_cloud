#!/bin/bash
set -e
set -x

source env.sh
### 0.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh 1 384

### 1.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh quant_mode shape_mutable batch_size
bash run.sh force_float32 true 1 384

### 2.infer
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh quant_mode shape_mutable batch_size max_seq_length
bash run.sh force_float32 true 32 384
