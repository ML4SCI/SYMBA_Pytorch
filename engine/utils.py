import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from engine.predictor import Predictor
PAD_IDX =  1

class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for (src_sample, tgt_sample) in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
        
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

def sequence_accuracy(config, device):
    predictor = Predictor(config, device)
    test_df = pd.read_csv('./data/'+config.dataset_name+'/test.csv')
    count = 0
    length = 500
    if config.debug:
        length = 10
    random_df = test_df.sample(n=length, random_state=config.seed)
    pbar = tqdm(range(length))
    pbar.set_description("Seq_Acc_Cal")
    for i in pbar:
        original_tokens, predicted_tokens = predictor.predict(random_df.iloc[i], raw_tokens=True)
        original_tokens = original_tokens.tolist()
        predicted_tokens = predicted_tokens.tolist()
        if original_tokens == predicted_tokens:
            count = count+1
        pbar.set_postfix(seq_accuracy=count/(i+1))
    return count/length
    
def init_distributed_mode(config, rank):
    # initialize the process group
    dist.init_process_group(backend=config.backend, rank = rank, world_size = config.world_size)
    torch.cuda.set_device(rank)
    
def cleanup():
    dist.destroy_process_group()
