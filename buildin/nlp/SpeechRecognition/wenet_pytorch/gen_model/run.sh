#!/bin/bash
magicmind_encoder_model=$1
magicmind_decoder_model=$2
precision=$3
batch_size=$4
dynamic_shape=$5
python $PROJ_ROOT_PATH/gen_model/gen_model.py --precision ${precision} \
					      --dynamic_shape ${dynamic_shape} \
                                              --magicmind_model $magicmind_encoder_model \
                                              --input_dims ${batch_size} 500 80 \
					      --input_dims ${batch_size} \
					      --onnx $MODEL_PATH/encoder.onnx \
                                              --type64to32_conversion true

python $PROJ_ROOT_PATH/gen_model/gen_model.py --precision ${precision} \
                                              --dynamic_shape ${dynamic_shape} \
                                              --magicmind_model $magicmind_decoder_model \
                                              --input_dims ${batch_size} 125 512 \
					      --input_dims ${batch_size} \
					      --input_dims ${batch_size} 4 24 \
					      --input_dims ${batch_size} 4 \
					      --input_dims ${batch_size} 4 \
					      --onnx $MODEL_PATH/decoder.onnx \
                                              --type64to32_conversion true
