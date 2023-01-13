#!/bin/bash
set -e
set -x

precision=force_float32
batch_size=1
w=200
language=infer_python
network=crnn_pytorch
### 0.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh

### 1.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
bash run.sh $precision

### 2 infer_python(including eval)
cd $PROJ_ROOT_PATH/infer_python
bash run.sh $precision $batch_size
