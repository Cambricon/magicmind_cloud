#!/bin/bash
magicmind_encoder_model=$1
magicmind_decoder_model=$2
magicmind_postnet_model=$3
magicmind_waveglow_model=$4
batch_size=$5

infer_res_dir="$PROJ_ROOT_PATH/data/output/"
if [ ! -d ${infer_res_dir} ];
then
    mkdir -p $infer_res_dir
fi

python infer.py  --encoder_magicmind $magicmind_encoder_model \
                 --decoder_magicmind $magicmind_decoder_model \
                 --postnet_magicmind $magicmind_postnet_model \
                 --waveglow_magicmind $magicmind_waveglow_model \
                 --batch_size $batch_size \
                 --device_id 0 
