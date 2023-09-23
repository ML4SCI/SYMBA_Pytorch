from dataclasses import dataclass, field, fields
from typing import Optional

@dataclass
class Config:
    experiment_name: Optional[str] = "default"
    root_dir: Optional[str] = "./"
    device: Optional[str] = "cuda"
    save_at_epochs: Optional[list] = field(default_factory=list)
    debug: Optional[bool] = False
        
    #training parameters
    epochs: Optional[int] = 60
    seed: Optional[int] = 42
    use_half_precision: Optional[bool] = True
        
    #distributed training parameters
    distributed: Optional[bool] = False
    backend: Optional[str] = 'nccl'
    world_size: Optional[int] = 8
        
    #data loader parameters
    train_shuffle: Optional[bool] = True
    test_shuffle: Optional[bool] = False
    training_batch_size: Optional[int] = 128
    test_batch_size: Optional[int] = 256
    num_workers: Optional[int] = 4
    pin_memory: Optional[bool] = True
        
    # scheduler parameters
    scheduler_type: Optional[str] = "none" # multi_step or none
    scheduler_gamma: Optional[float] = 0
    scheduler_milestones: Optional[list] = field(default_factory=list)

    # optimizer parameters
    optimizer_type: Optional[str] = "adam" # sgd or adam
    optimizer_lr: Optional[float] = 0.0001    
    optimizer_momentum: Optional[float] = 0.9
    optimizer_weight_decay: Optional[float] = 0.0
    optimizer_no_decay: Optional[list] = field(default_factory=list)
    clip_grad_norm: Optional[float] = -1

    # dataset parameters
    dataset_name: Optional[str] = "QCD_Amplitude" # QCD_Amplitude, QED_Amplitude, QCD_Feynman, QED_Feynman
    vocab_size: Optional[int] = 2246
    tokenizer_type: Optional[str] = "seq2seq"
    pretrained_tokenizer: Optional[bool] = False
    maximum_sequence_length: Optional[int] = 256 
        
    # Model Parameters
    model_name: Optional[str] = "seq2seq_transformer"
    embedding_size: Optional[int] = 512
    hidden_dim: Optional[int] = 512
    nhead: Optional[int] = 8
    num_encoder_layers: Optional[int] = 3
    num_decoder_layers: Optional[int] = 3
    dropout: Optional[int] = 0.1
    pretrain: Optional[bool] = False
        
    def print_config(self):
        print("="*50+"\nConfig\n"+"="*50)
        for field in fields(self):
            print(field.name.ljust(30), getattr(self, field.name))
        print("="*50)

    def save(self, root_dir):
        path = root_dir + "/config.txt"
        with open(path, "w") as f:
            f.write("="*50+"\nConfig\n"+"="*50 + "\n")
            for field in fields(self):
                f.write(field.name.ljust(30) + ": " + str(getattr(self, field.name)) + "\n")
            f.write("="*50)   
        
