from transformers import AutoModelForSequenceClassification
import torch
from datasets import load_dataset
import sys
import os

if __name__=="__main__":
    MODEL_NAME = 'textattack/bert-base-uncased-MRPC'
    PRJ_ROOT_PATH = sys.argv[1]
    BATCH_SIZE = 1
    MAX_SEQ_LENGTH = 128
    
    PYTORCH_MODEL_PATH = os.path.join(PRJ_ROOT_PATH,'data/models/traced.pt')
    if not os.path.exists(PYTORCH_MODEL_PATH):
        save_dir = os.path.join(PRJ_ROOT_PATH,'data/models')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,cache_dir=save_dir)
        torch_model.config.return_dict = False 
        
        input_ids = torch.randint(0, 1, (BATCH_SIZE, MAX_SEQ_LENGTH))
        attention_mask = torch.randint(0, 1, (BATCH_SIZE, MAX_SEQ_LENGTH))
        token_type_ids = torch.randint(0, 1, (BATCH_SIZE, MAX_SEQ_LENGTH))
        torch.jit.save(torch.jit.trace(torch_model, (input_ids, attention_mask, token_type_ids)), PYTORCH_MODEL_PATH)
        print("Export torch model to pt model sucess!")
    

    
    