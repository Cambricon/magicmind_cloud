import numpy as np
import collections
import sys

def rewrite_pad_value(input_data, valid_seq_length, max_seq_length):
    assert len(input_data) == max_seq_length
    for i in range(max_seq_length):
        if i >= valid_seq_length:
            input_data[i] = -float(sys.float_info.max)
    return input_data

def postprocess(results, features, max_seq_length):
    """ Split the raw result into individual results """
    single_result = []
    single_start_logits = []
    single_end_logits = []
    for result in results:
        start_logits = result['start_logits']
        end_logits = result['end_logits']
        rowsplit_start_logits = np.split(start_logits, start_logits.shape[0])
        rowsplit_end_logits = np.split(end_logits, end_logits.shape[0])
        single_start_logits.extend(rowsplit_start_logits)
        single_end_logits.extend(rowsplit_end_logits)
    assert len(features) == len(single_start_logits)
    for (feature_index, feature) in enumerate(features):
        unique_id = int(feature.unique_id)
        flat_start_logits = [float(x) for x in single_start_logits[feature_index].flat]
        flat_end_logits = [float(x) for x in single_end_logits[feature_index].flat]
        # rewrite pad value
        valid_seq_length = np.sum(feature.input_mask)
        flat_start_logits = rewrite_pad_value(flat_start_logits, valid_seq_length, max_seq_length)
        flat_end_logits = rewrite_pad_value(flat_end_logits, valid_seq_length, max_seq_length)
        single_result.append({
            'unique_ids': unique_id,
            'start_logits': flat_start_logits,
            'end_logits': flat_end_logits
        })
    return single_result
