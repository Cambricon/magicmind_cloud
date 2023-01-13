#!/bin/bash
set -e
set -x

### 0.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh

### 1.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh precision batch_size
bash run.sh force_float32 1

### 2.inference
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh  precision batch_size 
bash run.sh force_float32 1
