import torch
import argparse
import numpy as np
from scipy.io.wavfile import write
import magicmind.python.runtime as mm

import sys
import os
import os.path as path
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from text2sequence import prepare_input_sequence
sys.path.append("../")
from decoder import load_decoder_input_model, load_melcrop_model
from mm_runner import MMRunner
from logger import Logger

log = Logger()

def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--encoder_magicmind', required=True, help='encoder magicmind model')
    parser.add_argument('--decoder_magicmind', required=True, help='decoder magicmind model')
    parser.add_argument('--postnet_magicmind', required=True, help='postnet magicmind model')
    parser.add_argument('--waveglow_magicmind', required=True, help='waveglow magicmind model')
    parser.add_argument('-s', '--sigma-infer', default=0.6, type=float)
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('-il', '--input-length', type=int, default=128,
                        help='Input length')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')
    parser.add_argument('--device_id', type=int, default=0,
                        help='The device ordinal to be used. eg: "0,1,2"')
    parser.add_argument('--no-waveglow', default=False, action='store_true',
                        help='The waveglow model will not be tested when --no-waveglow is set.')
    args = parser.parse_args()
    print(args)
    return args

def infer_tacotron2_mm(device, encoder, decoder, postnet,
                       sequences, sequence_lengths, max_sequence_len):

    #encoder
    encoder_outputs = encoder([sequences, sequence_lengths])

    #decoder
    memory = encoder_outputs[0]
    processed_memory = encoder_outputs[1]
    memory_lengths = sequence_lengths
    max_memory_len = max_sequence_len
    
    bs = memory.shape[0]
    seq_len = memory.shape[1]
    attention_rnn_dim = 1024
    decoder_rnn_dim = 1024
    encoder_embedding_dim = 512
    n_mel_channels = 80

    tensor_shapes = [mm.Dims((bs, n_mel_channels)),         # decoder_input
                     mm.Dims((bs, attention_rnn_dim)),      # attention_hidden
                     mm.Dims((bs, attention_rnn_dim)),      # attention_cell
                     mm.Dims((bs, decoder_rnn_dim)),        # decoder_hidden
                     mm.Dims((bs, decoder_rnn_dim)),        # decoder_cell
                     mm.Dims((bs, seq_len)),                # attention_weights
                     mm.Dims((bs, seq_len)),                # attention_weights_cum
                     mm.Dims((bs, encoder_embedding_dim))]  # attention_context
    total_size = 0
    for shape in tensor_shapes:
        total_size += shape.GetElementCount()
    total_size *= 4  # for float32

    init_decoder_input_model = load_decoder_input_model(device)
    output_tensors = init_decoder_input_model([np.array([total_size], dtype=np.int32), memory_lengths, np.array([max_memory_len], dtype=np.int32)])
    mask = output_tensors[1]
    base_output = output_tensors[0]
    decoder_input_tensors = []
    start_index = 0
    for i in range(len(tensor_shapes)):
        end_index = start_index + tensor_shapes[i].GetElementCount()
        tensor=base_output[start_index:end_index]
        tensor=tensor.reshape(tensor_shapes[i].GetDims())
        start_index = end_index
        decoder_input_tensors.append(tensor)
    decoder_input_tensors.append(memory)
    decoder_input_tensors.append(processed_memory)
    decoder_input_tensors.append(mask)
    decoder_outputs = decoder(decoder_input_tensors)

    #postnet
    mel_outputs = decoder_outputs[0]
    melcrop_model = load_melcrop_model(device)
    mel_outputs = melcrop_model([mel_outputs, decoder_outputs[1]])[0]
    postnet_outputs = postnet([mel_outputs])

    mel_lengths = decoder_outputs[1]
    mel_outputs_postnet = postnet_outputs[0]
    return mel_outputs_postnet, mel_lengths

def infer_waveglow_mm(waveglow, mel):
    #waveglow
    mel_size = mel.shape[2]
    batch_size = mel.shape[0]
    stride = 256
    ngroup = 8
    z_size = mel_size * stride
    z_size = z_size // ngroup
    z = torch.randn(batch_size, ngroup, z_size).unsqueeze(3).numpy()
    mel = np.expand_dims(mel, 3)
    audios = waveglow([mel, z])
    return audios[0]

if __name__ == "__main__":
    args = get_args()

    if not os.path.exists(args.encoder_magicmind):
        log.info("please generate encoder model first!!!")
        exit()

    if not os.path.exists(args.decoder_magicmind):
        log.info("please generate decoder model first!!!")
        exit()

    if not os.path.exists(args.postnet_magicmind):
        log.info("please generate postnet model first!!!")
        exit()

    if not os.path.exists(args.waveglow_magicmind):
        log.info("please generate waveglow model first!!!")
        exit()

    # encoder 
    encoder = MMRunner(mm_file=args.encoder_magicmind, device_id=args.device_id)

    # decoder
    decoder = MMRunner(mm_file=args.decoder_magicmind, device_id=args.device_id)

    # postnet
    postnet = MMRunner(mm_file=args.postnet_magicmind, device_id=args.device_id)

    # waveglow
    waveglow = MMRunner(mm_file=args.waveglow_magicmind, device_id=args.device_id)

    texts = ["The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves. The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves."]
    texts = [texts[0][:args.input_length]]
    texts = texts * args.batch_size

    sequences, sequence_lengths = prepare_input_sequence(texts)
    batch_size = len(sequence_lengths)
    max_sequence_len = sequence_lengths[0]

    mel, mel_lengths = infer_tacotron2_mm(args.device_id, encoder, decoder, postnet, sequences, sequence_lengths, max_sequence_len)
    if not args.no_waveglow:
        audios = infer_waveglow_mm(waveglow, mel)

    num_mels = mel.shape[0] * mel.shape[2]

    if not args.no_waveglow:
        num_samples = audios.shape[0] * audios.shape[1]
        mel_lengths = mel_lengths.squeeze(1)
        #storage
        for i, audio in enumerate(audios):
            audio_path = "audio_"+str(i)+".wav"
            write(audio_path, args.sampling_rate,
                  audio[:mel_lengths[i]*args.stft_hop_length])
