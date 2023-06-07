#!/bin/bash
set -e
set -x
parameter_id=0
batch_size=4
precision=force_float32
shape_mutable=true
languages=infer_cpp

# ### 0.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh $parameter_id

### 1.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
bash run.sh $parameter_id $precision $shape_mutable $batch_size

### 2.infer and eval
if [ $languages == "infer_python" ];
then
    cd $PROJ_ROOT_PATH/infer_python
    bash run.sh $parameter_id $precision $shape_mutable
fi
if [ $languages == "infer_cpp" ];
then
    cd $PROJ_ROOT_PATH/infer_cpp
    bash run.sh $parameter_id $precision $shape_mutable
fi
