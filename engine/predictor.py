''' This Module help to predict the squared amplitude gievn the amplitude or feynman diagram'''
import torch
import pandas as pd
from torchtext.data import get_tokenizer
from models import get_model_from_config
from typing import Iterable, List
from torchtext.vocab import build_vocab_from_iterator
from transformers import AutoTokenizer

# Define special symbols and indices
BOS_IDX, PAD_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<s>', '<pad>', '</s>', '<unk>']

class Predictor:
    def __init__(self, config, device):
        self.model = get_model_from_config(config)
        self.dataset_name = config.dataset_name
        self.model_name = config.model_name
        self.path = './'+config.model_name+'/'+config.dataset_name+'/'+config.experiment_name+'/model_best.pth'
        self.device = config.device
        self.df = pd.read_csv('./data/'+config.dataset_name+'/train.csv')
        state = torch.load(self.path, map_location=self.device)
        self.model.load_state_dict(state['state_dict'])
        self.model.to(self.device)
        self.token_transform = get_tokenizer(tokenizer=None, language="en")
        self.tokenizer = None
        self.vocab_transform = {}
        self.text_transform = {}
        if "Amplitude" in config.dataset_name:
            self.l = ['Amplitude', 'Squared_Amplitude']
        else:
            self.l = ['Feynman_Diagram', 'Squared_Amplitude']
        for ln in self.l:
            self.vocab_transform[ln] = build_vocab_from_iterator(self.yeild_tokens(ln), specials=special_symbols, min_freq=1, 
                                                                 special_first=True, max_tokens=config.vocab_size)
            self.vocab_transform[ln].set_default_index(UNK_IDX)
            
            self.text_transform[ln] = self.sequential_transforms(self.token_transform, #Tokenization
                                                                 self.vocab_transform[ln], #Numericalization
                                                                 self.tensor_transform) # Add BOS/EOS and create tensor
        
    def yeild_tokens(self, language):
        for text in list(self.df[language]):
            yield self.token_transform(text)
    
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
    
    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)
        dim = 1
        
        if self.model_name != "seq2seq_transformer":
            memory = self.model.bart.model.encoder(torch.squeeze(src, 1))
        else:
            memory = self.model.encode(src, src_mask)
            memory = memory.to(self.device)
            dim = 0
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        for i in range(max_len-1):
            if self.model_name == "seq2seq_transformer":
                tgt_mask = (self.generate_square_subsequent_mask(ys.size(0), self.device).type(torch.bool)).to(self.device)
                out = self.model.decode(ys, memory, tgt_mask)
                out = out.transpose(0, 1)
                prob = self.model.generator(out[:, -1])
            else:
                out = self.model.bart.model.decoder(input_ids=ys, encoder_hidden_states=memory['last_hidden_state'])
                out = out[0]
                prob = self.model.bart.lm_head(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=dim)
            if next_word == EOS_IDX:
                break
        return ys
    
    def predict(self, test_example, raw_tokens=False):
        self.model.eval()
        src_sentence = test_example[self.l[0]]
        
        if self.model_name == "seq2seq_transformer":
            src = self.text_transform[self.l[0]](src_sentence).view(-1, 1)
            num_tokens = src.shape[0]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(f"data/{self.dataset_name}/{self.model_name}_tokenizer")
            src = self.tokenizer.encode(src_sentence, add_special_tokens=True, truncation=True, return_tensors="pt", max_length=512, padding="max_length")
            num_tokens = src.shape[1]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(src, src_mask, max_len=num_tokens+5, start_symbol=BOS_IDX).flatten()
        
        if raw_tokens:
            original_sentence = test_example[self.l[1]]
            if self.model_name == "seq2seq_transformer":
                original_tokens = self.text_transform[self.l[1]](original_sentence)
            else:
                original_tokens = self.tokenizer.encode(original_sentence, add_special_tokens=True, return_tensors="pt", padding="max_length")
            return original_tokens, tgt_tokens
        
        if self.model_name != "seq2seq_transformer":
            decoded_amplitude = self.tokenizer.decode(torch.squeeze(tgt_tokens, 0), skip_special_tokens=True)
        else:
            decoded_amplitude = " ".join(self.vocab_transform[self.l[1]].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<s>", "").replace("</s>", "")
        
        return decoded_amplitude
        
