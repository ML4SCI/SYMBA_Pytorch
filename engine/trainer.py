import os
import time
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils
from torch import autocast
from torch.cuda.amp import GradScaler
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from timeit import default_timer as timer
from engine.utils import *
from datasets import get_dataset_from_config
from models import get_model_from_config

class Trainer:
    def __init__(self, config, rank):
        self.config = config
        
        if self.config.distributed:
            self.device = rank
        else:
            self.device = torch.device(self.config.device)
        
        seed_everything(self.config.seed)
        
        # For distributed training 
        if self.config.distributed:
            init_distributed_mode(self.config, rank)
            cudnn.benchmark = True
            self.master_process = self.device == 0
        
        self.model_without_ddp = None
        self.model = self._prepare_model()
        
        # For half precision training 
        self.scaler = GradScaler()
        if self.config.use_half_precision:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        self.optimizer = self._prepare_optimizer()
        self.scheduler = self._prepare_scheduler()
        self.dataloaders = self._prepare_dataloaders()
        self.root_dir = None
        self._checks()
        self.stop_training = False
        
        self.current_epoch = 0
        self.best_accuracy = -12345
        self.best_val_loss = 1e6
        self.train_loss_list = []
        self.valid_loss_list = []
        self.valid_accuracy_tok_list = []
        
        if self.device==0 and self.config.distributed:
            self.config.print_config()
        elif not self.config.distributed:
            self.config.print_config()
        
    def criterion(self, y_pred, y_true):
        '''
        helper function to calculate loss
        '''
        pass
    
    def on_eval_end(self, valid_accuracy, valid_loss):
        '''
        helper function to excute at the end of evaluation
        '''
        pass
    
    def __call__(self, x):
        return self.model(x)

    def _prepare_model(self):
        model = get_model_from_config(self.config)
        model.to(self.device)
        self.model_without_ddp = model
        if self.config.distributed:
            model = DDP(model, device_ids=[self.device], find_unused_parameters=True)
        return model

    def _prepare_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        if self.config.distributed:
            param_optimizer = list(self.model.module.named_parameters())
        no_decay = self.config.optimizer_no_decay
        optimizer_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.config.optimizer_weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            ]

        if self.config.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(optimizer_parameters, lr=self.config.optimizer_lr, momentum=self.config.optimizer_momentum,)
        elif self.config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(optimizer_parameters, lr=self.config.optimizer_lr, eps=1e-9)
        else:
            raise NotImplementedError
        return optimizer

    def _prepare_scheduler(self):
        if self.config.scheduler_type == "multi_step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config.scheduler_milestones, gamma=self.config.scheduler_gamma)
        elif self.config.scheduler_type == "reduce_lr_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2)
        elif self.config.scheduler_type == "cosine_annealing_warm_restart":
            scheduler = torch.optim.lr_scheduler.CosineAnealingWarmRestarts(self.optimizer, T_0, T_mult)
        elif self.config.scheduler_type == "none":
            scheduler = None
        else:
            raise NotImplementedError
        return scheduler

    def _prepare_dataloaders(self):
        datasets=get_dataset_from_config(self.config)
        train_loader = torch.utils.data.DataLoader(datasets['train'], batch_size=self.config.training_batch_size, \
                                                   shuffle=self.config.train_shuffle, num_workers=self.config.num_workers, \
                                                   pin_memory=self.config.pin_memory, collate_fn=collate_fn)
        
        if self.config.distributed:
            sampler_train = torch.utils.data.DistributedSampler(datasets['train'], num_replicas=self.config.world_size, \
                                                                rank=self.device, shuffle=self.config.train_shuffle, seed=self.config.seed,)
            
            train_loader = torch.utils.data.DataLoader(datasets['train'], batch_size=self.config.training_batch_size, \
                                                   sampler=sampler_train, num_workers=self.config.num_workers, \
                                                   pin_memory=self.config.pin_memory, collate_fn=collate_fn)
        dataloaders = {
            'train': train_loader,
            'valid': torch.utils.data.DataLoader(datasets['valid'], \
                batch_size=self.config.test_batch_size, shuffle=self.config.test_shuffle, \
                    num_workers=self.config.num_workers, pin_memory=self.config.pin_memory, collate_fn=collate_fn),
            'test': torch.utils.data.DataLoader(datasets['test'], \
                batch_size=self.config.test_batch_size, shuffle=self.config.test_shuffle, \
                    num_workers=self.config.num_workers, pin_memory=self.config.pin_memory, collate_fn=collate_fn),
        }
        return dataloaders

    def set_root_dir(self, root_dir=None):
        if root_dir is None:
            root_dir = os.path.join(self.config.root_dir, \
                            self.config.model_name, \
                            self.config.dataset_name, \
                            self.config.experiment_name)
            if not self.config.distributed:
                print("==> Using default root directory for saving models:", root_dir)
            else:
                if self.device==0:
                    print("==> Using default root directory for saving models:", root_dir)

        self.root_dir = root_dir
        
    def load_best_model(self):
        file = os.path.join(self.root_dir, "model_best.pth")
        state = torch.load(file, map_location=self.device)
        if self.config.distributed:
            self.model = self.model_without_ddp
        self.model.load_state_dict(state['state_dict'])

    def _checks(self):
        if self.root_dir is None:
            self.set_root_dir()
        if os.path.exists(self.root_dir):
            if not self.config.distributed:
                print("==> Root directory already exists. Overwriting...")
            else:
                if self.device==0:
                    print("==> Root directory already exists. Overwriting...")
        os.makedirs(self.root_dir, exist_ok=True)
        self.config.save(self.root_dir)  
    
    def _train_epoch(self):
        self.model.train()
        disable = False
        if self.config.distributed:
            disable = not self.device==0
        pbar = tqdm(self.dataloaders['train'], total=len(self.dataloaders['train']), disable=disable)
        pbar.set_description(f"[{self.current_epoch+1}/{self.config.epochs}] Train")
        running_loss = AverageMeter()
        for src, tgt in pbar:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            bs = src.size(1)
            
            with autocast(device_type='cuda', dtype=self.dtype):
                if self.config.model_name == "seq2seq_transformer":
                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt[:-1, :], self.device)

                    logits = self.model(src, tgt[:-1, :], src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

                    # Calculate loss
                    loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt[1:, :].reshape(-1))
                else:
                    src = src.transpose(0, 1)
                    tgt = tgt.transpose(0, 1)
                    input_ids = torch.squeeze(src, 1)
                    target_ids = torch.squeeze(tgt, 1)[:, :-1]
                    outputs = self.model(input_ids, target_ids)
                    logits = outputs.logits

                    # Calculate loss
                    loss = self.criterion(logits.view(-1, logits.shape[-1]), torch.squeeze(tgt, 1)[:, 1:].reshape(-1))
                
            running_loss.update(loss.item(), bs)
            pbar.set_postfix(loss=running_loss.avg)
            
            # Backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.config.debug:
                break
                
            
        return running_loss.avg
    
    def evaluate(self, phase):
        self.model.eval()
        
        self.model.train()
        disable = False
        if self.config.distributed:
            disable = not self.device==0
        pbar = tqdm(self.dataloaders[phase], total=len(self.dataloaders[phase]), disable=disable)
        pbar.set_description(f"[{self.current_epoch+1}/{self.config.epochs}] {phase.capitalize()}")
        running_loss = AverageMeter()
        running_acc_tok = AverageMeter()
        
        with torch.no_grad():
            for src, tgt in pbar:
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                bs = src.size(1)
                
                if self.config.model_name == "seq2seq_transformer":
                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt[:-1, :], self.device)

                    logits = self.model(src, tgt[:-1, :], src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                    loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt[1:, :].reshape(-1))
                    y_pred = torch.argmax(logits.reshape(-1, logits.shape[-1]), 1)
                    correct = (y_pred == tgt[1:, :].reshape(-1)).cpu().numpy().mean()
                else:
                    src = src.transpose(0, 1)
                    tgt = tgt.transpose(0, 1)
                    input_ids = torch.squeeze(src, 1)
                    target_ids = torch.squeeze(tgt, 1)[:, :-1]
                    outputs = self.model(input_ids, target_ids)
                    logits = outputs.logits
                    
                    loss = self.criterion(logits.view(-1, logits.shape[-1]), torch.squeeze(tgt, 1)[:, 1:].reshape(-1))
                    y_pred = torch.argmax(logits.view(-1, logits.shape[-1]), 1)
                    correct = (y_pred == torch.squeeze(tgt, 1)[:, 1:].reshape(-1)).cpu().numpy().mean()
                    
                
                running_loss.update(loss.item(), bs)
                running_acc_tok.update(correct, bs)
                if self.config.distributed and self.device==0:
                    pbar.set_postfix(loss=running_loss.avg, tok_accuracy=running_acc_tok.avg)
                
                if self.config.debug:
                    break
                
        return running_acc_tok.avg, running_loss.avg
    
    def _save_model(self, checkpoint_name):
        state_dict = self.model.state_dict()
        if self.config.distributed:
            state_dict = self.model.module.state_dict()
        torch.save({
                "epoch": self.current_epoch + 1,
                "state_dict": state_dict,
                'optimizer': self.optimizer.state_dict(),
                "train_loss_list": self.train_loss_list,
                "valid_loss_list": self.valid_loss_list,
                "valid_accuracy_tok_list": self.valid_accuracy_tok_list,
            }, os.path.join(self.root_dir, checkpoint_name))
        
    def _save_and_log(self, accuracy):
        ## Save model
        if self.current_epoch + 1 in self.config.save_at_epochs:
            self._save_model(f"model_ep{self.current_epoch + 1 }.pth")
            print(f"==> Saved checkpoint {self.current_epoch + 1}")

        if not self.config.distributed:
            self._save_model("checkpoint.pth")
            if accuracy>self.best_accuracy:
                print(f"==> Best Accuracy improved to {round(accuracy, 4)} from {self.best_accuracy}")
                self.best_accuracy = round(accuracy, 4)
                shutil.copyfile(os.path.join(self.root_dir, "checkpoint.pth"), os.path.join(self.root_dir, "model_best.pth"))

        ## Log results
        data_list = [self.train_loss_list, self.valid_loss_list, self.valid_accuracy_tok_list]
        column_list = ['train_losses', 'valid_losses', 'token_valid_accuracy']
        
        df_data = np.array(data_list).T
        df = pd.DataFrame(df_data, columns=column_list)
        df.to_csv(os.path.join(self.root_dir, "logs.csv"))
        
        if self.config.distributed:
            self._save_model("checkpoint.pth")
            
    def _test_seq_acc(self):
        self.device = 'cuda'
        self.load_best_model()
        self.config.distributed = False
        test_accuracy_tok, _ = self.evaluate("test")
        test_accuracy_seq = sequence_accuracy(self.config, self.device)
        f= open(os.path.join(self.root_dir, "score.txt"),"w+")
        f.write(f"Token Accuracy = {(round(test_accuracy_tok, 4))}\n")
        f.write(f"Sequence Accuracy = {(round(test_accuracy_seq, 4))}\n")
        f.close()
        print(f"Test Accuracy: {round(test_accuracy_tok, 4)} | Valid Accuracy: {self.best_accuracy}")

    def fit(self):
        start_epoch = self.current_epoch
        for self.current_epoch in range(start_epoch, self.config.epochs):
            training_loss = self._train_epoch() 
            valid_accuracy_tok, valid_loss = self.evaluate("valid")
            
            self.train_loss_list.append(round(training_loss, 4))
            self.valid_loss_list.append(round(valid_loss, 4))
            self.valid_accuracy_tok_list.append(round(valid_accuracy_tok, 4))
            
            if self.scheduler == "multi_step":
                self.scheduler.step()
            elif self.scheduler == "reduce_lr_on_plateau":
                self.scheduler.step(valid_loss)
                
            if valid_loss<self.best_val_loss:
                self.best_val_loss = valid_loss

            if self.config.distributed:
                if self.master_process:
                    self._save_and_log(valid_accuracy_tok)
                    if valid_accuracy_tok>self.best_accuracy:
                        print(f"==> Best Accuracy improved to {round(valid_accuracy_tok, 4)} from {self.best_accuracy}")
                        self.best_accuracy = round(valid_accuracy_tok, 4)
                        self._save_model("model_best.pth")
            else:
                self._save_and_log(valid_accuracy_tok)
            
            self.on_eval_end(valid_accuracy_tok, valid_loss)

            if self.stop_training or self.config.debug:
                break
            
        if self.config.distributed and self.master_process:
            self._test_seq_acc()
            self.config.distributed = True
        elif not self.config.distributed:
            self._test_seq_acc()

        
        if self.config.distributed:
            cleanup()
