import magicmind.python.runtime as mm
import numpy as np
from mm_runner import MMRunner
# used to init decoder inputs
def load_decoder_input_model(device_id):
    dtype = mm.DataType.FLOAT32

    network = mm.Network()
    zero_const = network.add_i_const_node(dtype, mm.Dims([]), [0])
    zero_const_int = network.add_i_const_node(mm.DataType.INT32, mm.Dims([]), [0])
    one_const_1_dim = network.add_i_const_node(mm.DataType.INT32, mm.Dims([1]), [1])
    zero_const_1_dim = network.add_i_const_node(mm.DataType.INT32, mm.Dims([1]), [0])
    memory_size = network.add_input(mm.DataType.INT32, mm.Dims([1]))
    sequence_lengths = network.add_input(mm.DataType.INT32)
    max_sequence_len = network.add_input(mm.DataType.INT32, mm.Dims([1]))

    '''
    mask, implementation:

    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).byte()
    mask = torch.le(mask, 0)
    '''
    ids = network.add_i_range_node(zero_const_1_dim.get_output(0), max_sequence_len, one_const_1_dim.get_output(0))
    sequence_lengths = network.add_i_unsqueeze_node(sequence_lengths, one_const_1_dim.get_output(0))
    mask = network.add_i_logic_node(ids.get_output(0), sequence_lengths.get_output(0), mm.enums.ILogic.LT)
    mask = network.add_i_cast_node(mask.get_output(0), mm.DataType.INT32)
    mask = network.add_i_logic_node(mask.get_output(0), zero_const_int.get_output(0), mm.enums.ILogic.LE)

    # fill tensors with zero
    memory = network.add_i_fill_node(memory_size, zero_const.get_output(0))

    network.mark_output(memory.get_output(0))
    network.mark_output(mask.get_output(0))

    builder = mm.Builder()
    builder_config = mm.BuilderConfig()
    model = builder.build_model("decoder_input", network, builder_config)
    assert model is not None, 'build decoder input model failed'
    assert model.serialize_to_file("./decoder_input_model").ok()
    return MMRunner(mm_file="./decoder_input_model", device_id=device_id)

# mel_outputs = mel_outputs[:,:,:torch.max(mel_lengths)] mm implementation
# input1: mel_outputs from decoder
# input2: mel_lengths from decoder
# output: mels(postnet input)
def load_melcrop_model(device_id):
    network = mm.Network()
    mel_inputs = network.add_input(mm.DataType.FLOAT32)
    mel_lengths = network.add_input(mm.DataType.INT32)
    # torch.max(mel_lengths) # mm reduce max with axis=0
    max_axis = network.add_i_const_node(dtype=mm.DataType.INT32, dimensions=mm.Dims([1]), value=np.array([0], dtype=np.int32)).get_output(0)
    reduce_max = network.add_i_reduce_node(mel_lengths, max_axis, mm.IReduce.MAX, False).get_output(0)
    # mel_outputs[:,:,:max_len], mm strided slice with axis=2, stride=1
    begin_idx = max_axis  # 0
    slice_axis = network.add_i_const_node(dtype=mm.DataType.INT32, dimensions=mm.Dims([1]), value=np.array([2], dtype=np.int32)).get_output(0)
    mel_outputs = network.add_i_slice_node(mel_inputs, begin_idx, reduce_max, slice_axis).get_output(0)
    assert network.mark_output(mel_outputs).ok()

    # build
    config = mm.BuilderConfig()
    assert config.parse_from_string('{"precision_config":{"precision_mode":"force_float32"}}').ok()
    assert config.parse_from_string('{"archs":["mtp_372"]}').ok()
    builder = mm.Builder()
    assert builder is not None
    model = builder.build_model('melcrop', network, config)
    assert model is not None, 'build melcrop model failed'
    assert model.serialize_to_file("./melcrop_model").ok()
    return MMRunner(mm_file="./melcrop_model", device_id=device_id)
