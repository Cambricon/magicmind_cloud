#!/bin/bash
PRECISION=$1
BATCH_SIZE_MIN=$2
BATCH_SIZE=$3
BATCH_SIZE_MAX=$4
if [ -f $MODEL_PATH/encoder_${PRECISION}_${BATCH_SIZE_MIN}_${BATCH_SIZE_MAX}.graph ] && [ -f $MODEL_PATH/decoder_${PRECISION}_${BATCH_SIZE_MIN}_${BATCH_SIZE_MAX}.graph ] && [ -f $MODEL_PATH/postnet_${PRECISION}_${BATCH_SIZE_MIN}_${BATCH_SIZE_MAX}.graph ] && [ -f $MODEL_PATH/waveglow_${PRECISION}_${BATCH_SIZE_MIN}_${BATCH_SIZE_MAX}.graph ];
then
  echo "magicmind models already exist!!!"
else 
  echo "generate Magicmind model begin..."
  python $PROJ_ROOT_PATH/gen_model/gen_model.py -o $MODEL_PATH \
                                                --encoder $MODEL_PATH/onnx_models/encoder.onnx \
                                                --decoder $MODEL_PATH/onnx_models/decoder.onnx \
                                                --postnet $MODEL_PATH/onnx_models/postnet.onnx \
                                                --waveglow $MODEL_PATH/onnx_models/waveglow.onnx \
                                                --precision $PRECISION \
                                                -bs $BATCH_SIZE_MIN,$BATCH_SIZE,$BATCH_SIZE_MAX
fi
