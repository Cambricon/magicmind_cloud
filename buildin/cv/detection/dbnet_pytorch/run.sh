#!/bin/bash
set -e
set -x

precision=force_float32
shape_mutable=true
n=1
h=736
w=1280
network=dbnet_pytorch

### 0.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh

### 1.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
bash run.sh $precision $shape_mutable $n $h $w

### 2 infer_python(including eval)
cd $PROJ_ROOT_PATH/infer_python
bash run.sh $precision
