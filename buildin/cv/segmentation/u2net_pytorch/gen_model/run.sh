#!/bin/bash
PRECISION=$1
BATCH_SIZE=$2

if [ ! -d $MODEL_PATH ];
then
    mkdir -p $MODEL_PATH
fi
if [ ! -f $MODEL_PATH/u2net_pytorch_${PRECISION}_${BATCH_SIZE} ];
then
    echo "generate Magicmind model begin..."
    python $PROJ_ROOT_PATH/gen_model/gen_model.py --pt_model $MODEL_PATH/u2net.pt \
                                                  --output_model $MODEL_PATH/u2net_pytorch_${PRECISION}_${BATCH_SIZE} \
                                                  --precision ${PRECISION} \
                                                  --file_list $PROJ_ROOT_PATH/gen_model/file_list \
						  --batch_size ${BATCH_SIZE}
fi
