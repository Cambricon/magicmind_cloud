#!/bin/bash
magicmind_encoder_model=$1
magicmind_decoder_model=$2
magicmind_postnet_model=$3
magicmind_waveglow_model=$4
precision=$5
batch_size=$6
dynamic_shape=$7
#encoder
python $PROJ_ROOT_PATH/gen_model/gen_model.py --precision ${precision} \
                                              --magicmind_model ${magicmind_encoder_model} \
					      --onnx $MODEL_PATH/encoder.onnx \
					      --dynamic_shape ${dynamic_shape} \
                                              --input_dims ${batch_size} 128 \
					      --input_dims ${batch_size} \
                                              --dim_range_min 1 4 \
					      --dim_range_min 1 \
                                              --dim_range_max 16 256 \
                                              --dim_range_max 16 \
					      --type64to32_conversion true \
					      --conv_scale_fold true
#decoder
python $PROJ_ROOT_PATH/gen_model/gen_model.py --precision ${precision} \
                                              --magicmind_model ${magicmind_decoder_model} \
                                              --onnx $MODEL_PATH/decoder.onnx \
                                              --dynamic_shape ${dynamic_shape} \
                                              --input_dims ${batch_size} 80 \
                                              --input_dims ${batch_size} 1024 \
                                              --input_dims ${batch_size} 1024 \
                                              --input_dims ${batch_size} 1024 \
                                              --input_dims ${batch_size} 1024 \
                                              --input_dims ${batch_size} 128 \
                                              --input_dims ${batch_size} 128 \
                                              --input_dims ${batch_size} 512\
                                              --input_dims ${batch_size} 128 512 \
                                              --input_dims ${batch_size} 128 128 \
                                              --input_dims ${batch_size} 128 \
                                              --dim_range_min 1 80 \
                                              --dim_range_min 1 1024 \
                                              --dim_range_min 1 1024 \
                                              --dim_range_min 1 1024 \
                                              --dim_range_min 1 1024 \
                                              --dim_range_min 1 4 \
                                              --dim_range_min 1 4 \
                                              --dim_range_min 1 512 \
                                              --dim_range_min 1 4 512 \
                                              --dim_range_min 1 4 128 \
                                              --dim_range_min 1 4 \
                                              --dim_range_max 16 80 \
                                              --dim_range_max 16 1024 \
                                              --dim_range_max 16 1024 \
                                              --dim_range_max 16 1024 \
                                              --dim_range_max 16 1024 \
                                              --dim_range_max 16 256 \
                                              --dim_range_max 16 256 \
                                              --dim_range_max 16 512 \
                                              --dim_range_max 16 256 512 \
                                              --dim_range_max 16 256 128 \
                                              --dim_range_max 16 256 \
                                              --type64to32_conversion true \
                                              --conv_scale_fold true

#postnet
python $PROJ_ROOT_PATH/gen_model/gen_model.py --precision ${precision} \
                                              --magicmind_model ${magicmind_postnet_model} \
                                              --onnx $MODEL_PATH/postnet.onnx \
                                              --dynamic_shape ${dynamic_shape} \
                                              --input_dims ${batch_size} 80 512 \
                                              --dim_range_min 1 80 32 \
                                              --dim_range_max 16 80 1664 \
                                              --type64to32_conversion true \
                                              --conv_scale_fold true

#waveglow
python $PROJ_ROOT_PATH/gen_model/gen_model.py --precision ${precision} \
                                              --magicmind_model ${magicmind_waveglow_model} \
                                              --onnx $MODEL_PATH/waveglow.onnx \
                                              --dynamic_shape ${dynamic_shape} \
                                              --input_dims ${batch_size} 80 768 1 \
                                              --input_dims ${batch_size} 8 24576 1 \
                                              --dim_range_min 1 80 32 1 \
                                              --dim_range_min 1 8 1024 1 \
                                              --dim_range_max 16 80 1664 1 \
                                              --dim_range_max 16 8 53248 1 \
                                              --type64to32_conversion true \
                                              --conv_scale_fold true
