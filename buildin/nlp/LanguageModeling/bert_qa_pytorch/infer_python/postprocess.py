import torch
import transformers 
from transformers.data.processors.squad import SquadResult

def postprocess(features, example_indices, outputs):
    results = []
    for i, feature_index in enumerate(example_indices):
        eval_feature = features[feature_index.item()]
        unique_id = int(eval_feature.unique_id)
        output = [output[i].tolist() for output in outputs]
        start_logits, end_logits = output
        result = SquadResult(unique_id, start_logits, end_logits)
        results.append(result)
    return results