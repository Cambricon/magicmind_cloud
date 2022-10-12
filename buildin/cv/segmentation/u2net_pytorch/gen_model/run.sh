#!/bin/bash
QUANT_MODE=$1
BATCH_SIZE=$2

if [ ! -d $PROJ_ROOT_PATH/data/models/ ];
then
    mkdir $PROJ_ROOT_PATH/data/models/
fi
if [ ! -f $PROJ_ROOT_PATH/data/models/u2net_pytorch_${QUANT_MODE}_${BATCH_SIZE} ];
then
    echo "generate Magicmind model begin..."
    python $PROJ_ROOT_PATH/gen_model/gen_model.py --pt_model $MODEL_PATH/u2net.pt \
                                                  --output_model $PROJ_ROOT_PATH/data/models/u2net_pytorch_${QUANT_MODE}_${BATCH_SIZE} \
                                                  --quant_mode ${QUANT_MODE} \
                                                  --file_list $PROJ_ROOT_PATH/gen_model/file_list \
						  --batch_size ${BATCH_SIZE}
fi
