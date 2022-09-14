import numpy as np
import sys
sys.path.append("../../../../")
from thirdparty.text import text_to_sequence

# taken from tacotron2/data_function.py:TextMelCollate.__call__
def pad_sequences(texts):
    # Right zero-pad all one-hot text sequences to max input length
    ids = np.argsort(
        np.array([-len(x) for x in texts], dtype=np.int64), axis=0)
    max_input_len = texts[ids[0]].shape[0]

    text_padded = np.zeros(shape=(len(texts), max_input_len), dtype=np.int32)
    input_lengths = np.zeros(shape=ids.shape, dtype=np.int32)
    for i in range(len(ids)):
        text = texts[ids[i]]
        input_lengths[i] = text.shape[0]
        text_padded[i, :text.shape[0]] = text

    return text_padded, input_lengths

def prepare_input_sequence(texts):
    seqs = []
    for i,text in enumerate(texts):
        seqs.append(np.array(
            text_to_sequence(text, ['english_cleaners'])[:], dtype=np.int32))

    text_padded, input_lengths = pad_sequences(seqs)

    return text_padded, input_lengths

def test():
    texts = ["The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves. The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves."]
    text_padded, _ = prepare_input_sequence(texts)

    # dump from trt
    expect = np.array([[57, 45, 42, 11, 43, 52, 55, 50, 56, 11, 52, 43, 11, 53, 55, 46, 51, 57,
        42, 41, 11, 49, 42, 57, 57, 42, 55, 56, 11, 56, 45, 52, 58, 49, 41, 11,
        39, 42, 11, 39, 42, 38, 58, 57, 46, 43, 58, 49,  6, 11, 38, 51, 41, 11,
        57, 45, 38, 57, 11, 57, 45, 42, 46, 55, 11, 38, 55, 55, 38, 51, 44, 42,
        50, 42, 51, 57, 11, 52, 51, 11, 57, 45, 42, 11, 53, 38, 44, 42, 11, 56,
        45, 52, 58, 49, 41, 11, 39, 42, 11, 55, 42, 38, 56, 52, 51, 38, 39, 49,
        42, 11, 38, 51, 41, 11, 38, 11, 45, 42, 49, 53, 11, 57, 52, 11, 57, 45,
        42, 11, 56, 45, 38, 53, 42, 49, 46, 51, 42, 56, 56, 11, 52, 43, 11, 57,
        45, 42, 11, 49, 42, 57, 57, 42, 55, 56, 11, 57, 45, 42, 50, 56, 42, 49,
        59, 42, 56,  7, 11, 57, 45, 42, 11, 43, 52, 55, 50, 56, 11, 52, 43, 11,
        53, 55, 46, 51, 57, 42, 41, 11, 49, 42, 57, 57, 42, 55, 56, 11, 56, 45,
        52, 58, 49, 41, 11, 39, 42, 11, 39, 42, 38, 58, 57, 46, 43, 58, 49,  6,
        11, 38, 51, 41, 11, 57, 45, 38, 57, 11, 57, 45, 42, 46, 55, 11, 38, 55,
        55, 38, 51, 44, 42, 50, 42, 51, 57, 11, 52, 51, 11, 57, 45, 42, 11, 53,
        38, 44, 42, 11, 56, 45, 52, 58, 49, 41, 11, 39, 42, 11, 55, 42, 38, 56,
        52, 51, 38, 39, 49, 42, 11, 38, 51, 41, 11, 38, 11, 45, 42, 49, 53, 11,
        57, 52, 11, 57, 45, 42, 11, 56, 45, 38, 53, 42, 49, 46, 51, 42, 56, 56,
        11, 52, 43, 11, 57, 45, 42, 11, 49, 42, 57, 57, 42, 55, 56, 11, 57, 45,
        42, 50, 56, 42, 49, 59, 42, 56,  7]], dtype=np.int32)

    assert (text_padded == expect).all()
    print("Success!")

if __name__ == '__main__':
    test()

