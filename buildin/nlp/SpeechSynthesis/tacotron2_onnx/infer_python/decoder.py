import magicmind.python.runtime as mm
import numpy as np
import sys
sys.path.append("../")
from gen_model.mm_utils import NNModel, _check_status

# used to init decoder inputs
def load_decoder_input_model(device : mm.Device):
    dtype = mm.DataType.FLOAT32

    network = mm.Network()
    zero_const = network.add_i_const_node(dtype, mm.Dims([]), [0])
    zero_const_int = network.add_i_const_node(mm.DataType.INT32, mm.Dims([]), [0])
    one_const_1_dim = network.add_i_const_node(mm.DataType.INT32, mm.Dims([1]), [1])
    zero_const_1_dim = network.add_i_const_node(mm.DataType.INT32, mm.Dims([1]), [0])
    memory_size = network.add_input(mm.DataType.INT32, mm.Dims([]))
    sequence_lengths = network.add_input(mm.DataType.INT32)
    max_sequence_len = network.add_input(mm.DataType.INT32, mm.Dims([]))

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
    return NNModel(device, model)

# mel_outputs = mel_outputs[:,:,:torch.max(mel_lengths)] mm implementation
# input1: mel_outputs from decoder
# input2: mel_lengths from decoder
# output: mels(postnet input)
def load_melcrop_model(device : mm.Device):
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
    return NNModel(device, model)

class Decoder(NNModel):
    def __init__(self, device : mm.Device, mm_model : mm.Model):
        super(Decoder, self).__init__(device, mm_model)
        self.decoder_input_model = load_decoder_input_model(device)
        self.melcrop_model = load_melcrop_model(device)

    def init_decoder_inputs(self, memory, processed_memory, memory_lengths, max_memory_len, measurements):
        
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

        input_tensors = self.decoder_input_model.get_raw_input_tensors()
        # _check_status(input_tensors[0].from_numpy(np.array([total_size], dtype=np.int32)))
        _check_status(input_tensors[0].memcpy_from_host(np.array([total_size], dtype=np.int32)))
        input_tensors[1] = memory_lengths
        # _check_status(input_tensors[2].from_numpy(np.array([max_memory_len], dtype=np.int32)))
        _check_status(input_tensors[2].memcpy_from_host(np.array([max_memory_len], dtype=np.int32)))

        output_tensors = self.decoder_input_model.execute(input_tensors)

        mask = output_tensors[1]
        decoder_input_tensors = self.get_raw_input_tensors()
        base_addr = output_tensors[0].data_address()
        for i in range(len(tensor_shapes)):
            decoder_input_tensors[i].set_data_address(base_addr, tensor_shapes[i].GetDims())
            base_addr += (tensor_shapes[i].GetElementCount() * 4)
        decoder_input_tensors[8] = memory
        decoder_input_tensors[9] = processed_memory
        decoder_input_tensors[10] = mask

        # hold output_tensors before del decoder_input_tensors(memory in decoder_input_tensors was allocated by output_tensors)
        self.init_model_outputs = output_tensors
        return decoder_input_tensors

    def melcrop(self, mel, mel_lengths):
        return self.melcrop_model.execute([mel, mel_lengths])[0]

def load_decoder(device : mm.Device, path : str):
    ''' load model from disk '''
    model = mm.Model()
    _check_status(model.deserialize_from_file(path))
    return Decoder(device, model)


