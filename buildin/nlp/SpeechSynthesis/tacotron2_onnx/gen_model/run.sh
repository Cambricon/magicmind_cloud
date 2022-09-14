#!/bin/bash
QUANT_MODE=$1 #force_float32/force_float16
BATCH_SIZE_MIN=$2
BATCH_SIZE=$3
BATCH_SIZE_MAX=$4

if [ -f $PROJ_ROOT_PATH/data/models/encoder_${QUANT_MODE}_model ] && [ -f $PROJ_ROOT_PATH/data/models/decoder_${QUANT_MODE}_model ] && [ -f $PROJ_ROOT_PATH/data/models/postnet_${QUANT_MODE}_model ] && [ -f $PROJ_ROOT_PATH/data/models/waveglow_${QUANT_MODE}_model ];
then
  echo "magicmind models already exist!!!"
else 
  echo "generate Magicmind model begin..."
    python $PROJ_ROOT_PATH/gen_model/gen_model.py -o $PROJ_ROOT_PATH/data/models \
                                                  --json $PROJ_ROOT_PATH/data/jsons/builder_config.json \
                                                  --encoder $PROJ_ROOT_PATH/data/models/encoder.onnx \
                                                  --decoder $PROJ_ROOT_PATH/data/models/decoder.onnx \
                                                  --postnet $PROJ_ROOT_PATH/data/models/postnet.onnx \
                                                  --waveglow $PROJ_ROOT_PATH/data/models/waveglow.onnx \
                                                  --quant_mode ${QUANT_MODE} \
                                                  -bs ${BATCH_SIZE_MIN},${BATCH_SIZE},${BATCH_SIZE_MAX}
fi
