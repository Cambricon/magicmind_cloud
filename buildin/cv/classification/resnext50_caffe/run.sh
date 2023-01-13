#!/bin/bash
set -e
set -x

### 0.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh

### 1.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh precision shape_mutable batch_size
bash run.sh force_float32 true 1

#### 2.infer_python
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh precision shape_mutable batch_size image_num
bash run.sh force_float32 true 1 100

### 2.2 infer_cpp
cd $PROJ_ROOT_PATH/infer_cpp
#bash run.sh precision shape_mutable batch_size image_num
bash run.sh force_float32 true 1 100
