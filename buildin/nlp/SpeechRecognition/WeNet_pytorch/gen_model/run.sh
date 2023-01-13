#!/bin/bash
PRECISION=$1 #force_float32

if [ -f $PROJ_ROOT_PATH/data/models/encoder_${PRECISION}_model ] && [ -f $PROJ_ROOT_PATH/data/models/decoder_${PRECISION}_model ];
then
  echo "magicmind models already exist!!!"
else 
  echo "generate Magicmind model begin..."
    python $PROJ_ROOT_PATH/gen_model/gen_model.py --output $PROJ_ROOT_PATH/data/models \
                                                  --json $PROJ_ROOT_PATH/data/jsons/builder_config.json \
                                                  --encoder $PROJ_ROOT_PATH/data/models/20211025_conformer_exp/onnx_model/encoder.onnx \
                                                  --decoder $PROJ_ROOT_PATH/data/models/20211025_conformer_exp/onnx_model/decoder.onnx \
                                                  --precision ${PRECISION} 
fi
