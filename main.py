
from typing import Union
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataloaders.loader import get_loader
from argparse import Namespace
import pdb
import torch as tc
import functools
from tqdm import tqdm 

def get_model_and_dataloader(model_name: str, dataset_name: str, split: str = "test", batch_size:int = 512, num_exs: Union[int, None] = None):
    '''
    Args:
        model_name: name of trained model to load. See https://huggingface.co/connectivity for options.
        dataset_name: name of dataset. See dataloaders/loader.py for options.
        split: name of data split. See dataloaders/get_datasets.py for options. `"train"` and `"test"` are generally valid options.
        num_exs: number of examples in the dataset. `None` represents full dataset.
    '''
    is_feather_bert = "feather" in model_name 
    if is_feather_bert:
        mnli_label_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}
    else:        
        mnli_label_dict = {"contradiction": 2, "entailment": 0, "neutral": 1}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(X):
            return func(**X)
        return wrapper
    
    model.forward = decorator(model.forward)

    dataloader = get_loader(Namespace(
        dataset = dataset_name , 
        split = split , 
        batch_size = batch_size , 
        num_exs = num_exs,  
    ), tokenizer = tokenizer, mnli_label_dict = mnli_label_dict)

    return model, dataloader

if __name__ == "__main__":

    model_name   = "connectivity/cola_6ep_ft-21"
    dataset_name = "qqp"
    model, dataloader = get_model_and_dataloader(model_name, dataset_name, num_exs = 5000)

    with tc.no_grad():
        for i, (input, target) in tqdm( enumerate(dataloader) ):
            output = model(input)
            preds = tc.max(output["logits"], dim=-1).indices
            pdb.set_trace()
