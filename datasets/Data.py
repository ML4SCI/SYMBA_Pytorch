import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from datasets.utils import create_csv_json, create_vocab
import torchtext
from torchtext.data import get_tokenizer
from transformers import AutoTokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List

# Define special symbols and indices
BOS_IDX, PAD_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<s>', '<pad>', '</s>', '<unk>'] 

class Transform:
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.token_transform = {}
        self.vocab_transform = {}
        self.text_transform = {}
        if "Amplitude" in self.config.dataset_name:
            self.l = ['Amplitude', 'Squared_Amplitude']
        else:
            self.l = ['Feynman_Diagram', 'Squared_Amplitude']
        
        for ln in self.l:
            self.token_transform[ln] = get_tokenizer(tokenizer=None, language="en")
            self.vocab_transform[ln] = build_vocab_from_iterator(self.yeild_tokens(ln), specials=special_symbols, min_freq=1, 
                                                                 special_first=True, max_tokens=self.vocab_size)
            self.vocab_transform[ln].set_default_index(UNK_IDX)
            
    def yeild_tokens(self, language):
        for text in list(self.df[language]):
            yield self.token_transform[language](text)
            
    # helper function to club together sequential operations
    def sequential_transforms(self, *transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func
        
    # function to add BOS/EOS and create tensor for input sequence indices
    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([BOS_IDX]),
                          torch.tensor(token_ids),
                          torch.tensor([EOS_IDX])))
    
    def adapt(self):
        for ln in self.l:
            self.text_transform[ln] = self.sequential_transforms(self.token_transform[ln], #Tokenization
                                                                 self.vocab_transform[ln], #Numericalization
                                                                 self.tensor_transform) # Add BOS/EOS and create tensor
    def transform(self, data):
        return (self.text_transform[self.l[0]](data[self.l[0]]), 
                self.text_transform[self.l[1]](data[self.l[1]]))
    
def get_transform(config):
    path = 'data/'+config.dataset_name+'/train.csv'
    df = pd.DataFrame()
    if os.path.exists(path):
        df = pd.read_csv(path)
        print("==> Using precomputed data-splits")
    else:
        create_csv_json('data/'+config.dataset_name+'/'+config.dataset_name.lower()+'_order_data.txt')
        df = pd.read_csv(path)
    trans_form = Transform(df, config)
    trans_form.adapt()
    return trans_form
    
class Dataset(nn.Module):
    def __init__(self, split, config, trans_form):
        self.transform = trans_form
        self.config = config
        self.path = 'data/'+self.config.dataset_name
        self.split = '/' + split + '.csv'
        self.df = pd.DataFrame() 
        self.df = pd.read_csv(self.path+self.split)
        if "Amplitude" in self.config.dataset_name:
            self.l = ['Amplitude', 'Squared_Amplitude']
        else:
            self.l = ['Feynman_Diagram', 'Squared_Amplitude']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.config.tokenizer_type == "seq2seq":
            return self.transform.transform(self.df.iloc[idx])
        else:
            ampl_tokens = self.transform.encode(self.df.iloc[idx][self.l[0]])
            sq_ampl_tokens = self.transform.encode(self.df.iloc[idx][self.l[1]])
        
            return (torch.tensor(ampl_tokens), torch.tensor(sq_ampl_tokens))
    
    @staticmethod
    def is_valid_dataset_name(dataset_name):
        if dataset_name in ['QED_Amplitude', 'QCD_Amplitude', 'QED_Feynman', 'QCD_Feynman']:
            return True
        
        return False
    
    @staticmethod
    def get_dataset_from_config(config):
        dataset_name = config.dataset_name
        tokenizer_path = ""
        
        if not Dataset.is_valid_dataset_name(dataset_name):
            raise ValueError('Invalid dataset: {}'.format(model_name))
            
        if config.tokenizer_type == "seq2seq":
            trns_form = get_transform(config)
            
        elif config.model_name in ['bart-base', 'bart-large']:
            if config.pretrained_tokenizer:
                trns_form = AutoTokenizer.from_pretrained(f'facebook/{config.model_name}')
            else:
                vocab_file = 'data/'+config.dataset_name+'/'+config.model_name+"_tokenizer"
                if not os.path.exists(vocab_file):
                    create_vocab(config)
                trns_form = AutoTokenizer.from_pretrained(vocab_file)
                
        elif config.model_name in ['LED-base', 'LED-large']:
            if config.pretrained_tokenizer:
                trns_form = AutoTokenizer.from_pretrained("allenai/led-base-16384")
            else:
                vocab_file = 'data/'+config.dataset_name+'/'+config.model_name+"_tokenizer"
                if not os.path.exists(vocab_file):
                    create_vocab(config)
                trns_form = AutoTokenizer.from_pretrained(vocab_file)
            
        train_dataset = Dataset('train', config, trns_form)
        val_dataset = Dataset('val', config, trns_form)
        test_dataset = Dataset('test', config, trns_form)
        
        return {"train": train_dataset, "valid": val_dataset, "test": test_dataset}
        
