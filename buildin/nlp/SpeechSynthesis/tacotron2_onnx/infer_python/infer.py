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
from gen_model.mm_utils import load_model, ndarray2mlu, _check_status, MeasureTime, parse_dynamic_size
from decoder import load_decoder

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Directory contains magicmind models(decoder, encoder, postnet, waveglow)')
    parser.add_argument('-s', '--sigma-infer', default=0.6, type=float)
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--num-iters', type=int, default=10,
                        help='Number of iterations')
    parser.add_argument('--warmup-iters', type=int, default=3,
                        help='Number of warmup iterations')
    parser.add_argument('-il', '--input-length', type=int, default=64,
                        help='Input length')
    parser.add_argument('-bs', '--batch-size', type=str, default="1",
                        help='One or three comma separated integers specifying the batch size. Specify "min,opt,max" for dynamic shape')
    parser.add_argument('--precision', type=str, default="force_float16",
                        help='float16 models or float32 models will be used.')
    parser.add_argument('--log_file', type=str, default='',
                        help='Benchmark log file, empty means log to std')
    parser.add_argument('--devices', type=str, default="0",
                        help='The devices ordinal to be used. eg: "0,1,2"')
    parser.add_argument('--no-waveglow', default=False, action='store_true',
                        help='The waveglow model will not be tested when --no-waveglow is set.')

    return parser

def load_models(device : mm.Device, models_dir : str, precision_str : str, bs_min : int, bs_max : int):
    encoder_model_path = path.join(models_dir, 'encoder_' + precision_str + "_" + str(bs_min) + "_" + str(bs_max) + '.graph')
    decoder_model_path = path.join(models_dir, 'decoder_' + precision_str + "_" + str(bs_min) + "_" + str(bs_max) + '.graph')
    postnet_model_path = path.join(models_dir, 'postnet_' + precision_str + "_" + str(bs_min) + "_" + str(bs_max) + '.graph')
    waveglow_model_path = path.join(models_dir, 'waveglow_' + precision_str + "_" + str(bs_min) + "_" + str(bs_max) + '.graph')
    return load_model(device, encoder_model_path), load_decoder(device, decoder_model_path), \
        load_model(device, postnet_model_path), load_model(device, waveglow_model_path)

def infer_tacotron2_mm(encoder, decoder, postnet,
                       sequences, sequence_lengths, max_sequence_len, measurements):

    #encoder
    encoder_inputs = encoder.get_raw_input_tensors()
    _check_status(encoder_inputs[0].from_numpy(sequences))
    encoder_inputs[1] = sequence_lengths
    with MeasureTime(measurements, "tacotron2_encoder_time"):
        encoder_outputs = encoder.execute(encoder_inputs)

    #decoder
    memory = encoder_outputs[0]
    processed_memory = encoder_outputs[1]
    decoder_inputs = decoder.init_decoder_inputs(memory, processed_memory,
            sequence_lengths, max_sequence_len, measurements)
    with MeasureTime(measurements, "tacotron2_decoder_time"):
        decoder_outputs = decoder.execute(decoder_inputs)

    #postnet
    postnet_inputs = postnet.get_raw_input_tensors()
    mel_outputs = decoder_outputs[0]
    mel_outputs = decoder.melcrop(mel_outputs, decoder_outputs[1])
    postnet_inputs[0] = mel_outputs
    with MeasureTime(measurements, "tacotron2_postnet_time"):
        mel_outputs_postnet = postnet.execute(postnet_inputs)[0]

    mel_lengths = decoder_outputs[1]
    return mel_outputs_postnet, mel_lengths

def infer_waveglow_mm(waveglow, mel, measurements):
    #waveglow
    mel_size = mel.shape[2]
    batch_size = mel.shape[0]
    stride = 256
    ngroup = 8
    z_size = mel_size * stride
    z_size = z_size // ngroup
    z = torch.randn(batch_size, ngroup, z_size).unsqueeze(3).numpy()
    inputs = waveglow.get_raw_input_tensors()
    _check_status(inputs[1].from_numpy(z))
    mel.reshape((mel.shape[0], mel.shape[1], mel.shape[2], 1))  # unsqueeze(3)
    inputs[0] = mel
    with MeasureTime(measurements, "waveglow_time"):
        audios = waveglow.execute(inputs)[0]
    return audios

def print_stats(measurements_all, log_file, no_waveglow):
    throughput = measurements_all['throughput']
    preprocessing = measurements_all['pre_processing']
    type_conversion = measurements_all['type_conversion']
    storage = measurements_all['storage']
    data_transfer = measurements_all['data_transfer']
    postprocessing = [sum(p) for p in zip(type_conversion,storage,data_transfer)]
    latency = measurements_all['latency']
    num_mels_per_audio = measurements_all['num_mels_per_audio']
    encoder_latency = measurements_all['tacotron2_encoder_time']
    decoder_latency = measurements_all['tacotron2_decoder_time']
    postnet_latency = measurements_all['tacotron2_postnet_time']
    waveglow_latency = measurements_all['waveglow_time']
    tacotron2_items_per_sec = measurements_all['tacotron2_items_per_sec']

    latency.sort()

    cf_50 = max(latency[:int(len(latency)*0.50)])
    cf_90 = max(latency[:int(len(latency)*0.90)])
    cf_95 = max(latency[:int(len(latency)*0.95)])
    cf_99 = max(latency[:int(len(latency)*0.99)])
    cf_100 = max(latency[:int(len(latency)*1.0)])

    if not no_waveglow:
        print("Throughput average (samples/sec) = {:.6f}".format(np.mean(throughput)), file=log_file)
    print("Preprocessing average (seconds) = {:.6f}".format(np.mean(preprocessing)), file=log_file)
    if not no_waveglow:
        print("Postprocessing average (seconds) = {:.6f}".format(np.mean(postprocessing)), file=log_file)
    print("Number of mels per audio average = {}".format(np.mean(num_mels_per_audio)), file=log_file)
    print("Encoder latency average (seconds) = {:.6f}".format(np.mean(encoder_latency)), file=log_file)
    print("Decoder latency average (seconds) = {:.6f}".format(np.mean(decoder_latency)), file=log_file)
    print("Postnet latency average (seconds) = {:.6f}".format(np.mean(postnet_latency)), file=log_file)
    if not no_waveglow:
        print("Waveglow latency average (seconds) = {:.6f}".format(np.mean(waveglow_latency)), file=log_file)
    print("Latency average (seconds) = {:.6f}".format(np.mean(latency)), file=log_file)
    print("Latency std (seconds) = {:.6f}".format(np.std(latency)), file=log_file)
    print("Latency cl 50 (seconds) = {:.6f}".format(cf_50), file=log_file)
    print("Latency cl 90 (seconds) = {:.6f}".format(cf_90), file=log_file)
    print("Latency cl 95 (seconds) = {:.6f}".format(cf_95), file=log_file)
    print("Latency cl 99 (seconds) = {:.6f}".format(cf_99), file=log_file)
    print("Latency cl 100 (seconds) = {:.6f}".format(cf_100), file=log_file)
    print("tacotron2_items_per_sec average = {:.6f}".format(np.mean(tacotron2_items_per_sec)), file=log_file)

def run(args, device, pipe):
    measurements_all = {"pre_processing": [],
                        "tacotron2_encoder_time": [],
                        "tacotron2_decoder_time": [],
                        "tacotron2_postnet_time": [],
                        "waveglow_time": [],
                        "tacotron2_latency": [],
                        "waveglow_latency": [],
                        "latency": [],
                        "type_conversion": [],
                        "data_transfer": [],
                        "storage": [],
                        "tacotron2_items_per_sec": [],
                        "waveglow_items_per_sec": [],
                        "num_mels_per_audio": [],
                        "throughput": []}
    # init magicmind
    mmsys = mm.System()
    mmsys.initialize()
    mmdev = mm.Device()
    mmdev.id = device
    _check_status(mmdev.active())

    # load MM models
    bs_min, bs_opt, bs_max = parse_dynamic_size(args.batch_size)
    encoder, decoder, postnet, waveglow = load_models(mmdev, args.models_dir, args.precision, bs_min, bs_max)

    texts = ["The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves. The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves."]
    texts = [texts[0][:args.input_length]]
    texts = texts * bs_opt

    for iter in range(args.num_iters + args.warmup_iters):
        measurements = {}

        with MeasureTime(measurements, "pre_processing"):
            sequences, sequence_lengths = prepare_input_sequence(texts)
            batch_size = len(sequence_lengths)
            max_sequence_len = sequence_lengths[0]
            # convert ndarray to mm tensor
            sequence_lengths_tensor = ndarray2mlu(sequence_lengths)

        with MeasureTime(measurements, "latency"):
            with MeasureTime(measurements, "tacotron2_latency"):
                mel, mel_lengths = infer_tacotron2_mm(encoder, decoder, postnet,
                                                      sequences, sequence_lengths_tensor, max_sequence_len,
                                                      measurements)
            if not args.no_waveglow:
                with MeasureTime(measurements, "waveglow_latency"):
                    audios = infer_waveglow_mm(waveglow, mel, measurements)

        num_mels = mel.shape[0] * mel.shape[2]

        if not args.no_waveglow:
            num_samples = audios.shape[0] * audios.shape[1]

            with MeasureTime(measurements, "type_conversion"):
                audios = audios.to(mm.DataType.FLOAT32)

            with MeasureTime(measurements, "data_transfer"):
                audios = audios.asnumpy()
                mel_lengths = mel_lengths.asnumpy().squeeze(1)

            with MeasureTime(measurements, "storage"):
                for i, audio in enumerate(audios):
                    audio_path = "audio_"+str(i)+".wav"
                    write(audio_path, args.sampling_rate,
                          audio[:mel_lengths[i]*args.stft_hop_length])

        if iter >= args.warmup_iters:
            measurements['tacotron2_items_per_sec'] = num_mels / measurements['tacotron2_latency']
            measurements['num_mels_per_audio'] = mel.shape[2]
            if not args.no_waveglow:
                measurements['waveglow_items_per_sec'] = num_samples / measurements['waveglow_latency']
                measurements['throughput'] = num_samples / measurements['latency']
            for k,v in measurements.items():
                if k in measurements_all.keys():
                    measurements_all[k].append(v)

        # update tqdm
        pipe[1].send(iter + 1)

    mmsys.destory()
    pipe[1].send(measurements_all)
    for p in pipe:
        p.close()

def main():
    parser = argparse.ArgumentParser(
        description='MagicMind Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    import re
    devices = re.split('\ |,', args.devices)
    if '' in devices:
        devices.remove('')
    print('Run on devices:', devices)
    import multiprocess
    pipes = []
    jobs = []
    for dev in devices:
        pipe = multiprocess.Pipe(False)
        pipes.append(pipe)
        p = multiprocess.Process(target=run, args=(args, int(dev), pipe))
        jobs.append(p)

    # start all
    for job in jobs:
        job.start()

    for pipe in pipes:
        pipe[1].close()

    # update tqdm
    expected_iters = len(devices) * (args.num_iters + args.warmup_iters)
    from tqdm import tqdm
    bar = tqdm(total=expected_iters, desc='running')
    last_iters = 0
    erased = []
    while True:
        cur_iters = 0
        for i in range(len(pipes)):
            if i in erased:
                continue
            out_pipe = pipes[i][0]
            dev_iters = out_pipe.recv()
            cur_iters += dev_iters
            if (args.num_iters + args.warmup_iters) == dev_iters:
                erased.append(i)
        bar.update(cur_iters - last_iters)
        last_iters = cur_iters
        if cur_iters == expected_iters:
            break
    bar.close()

    # wait all
    for job in jobs:
        job.join()

    
    # log profiles
    log_file = sys.stdout if len(args.log_file) == 0 else open(args.log_file, 'w')
    print("log_file: ", args.log_file)
    for i in range(len(devices)):
        out_pipe = pipes[i][0]
        measurements = out_pipe.recv()
        print(f'================dev[{devices[i]}]================', file=log_file)
        print_stats(measurements, log_file, args.no_waveglow)
    if log_file is not sys.stdout:
        log_file.close()

    for pipe in pipes:
        pipe[0].close()

if __name__ == '__main__':
    main()

