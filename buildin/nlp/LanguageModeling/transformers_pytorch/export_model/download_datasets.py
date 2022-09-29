from transformers import AutoModelForSequenceClassification
import torch
from datasets import load_dataset
import sys
import os

if __name__=="__main__":
    MODEL_NAME = 'textattack/bert-base-uncased-MRPC'
    DATASETS_PATH = sys.argv[1]
    BATCH_SIZE = 1
    MAX_SEQ_LENGTH = 128
    # download dataset
    raw_datasets = load_dataset('glue', 'mrpc', cache_dir=os.path.join(DATASETS_PATH))

    
    