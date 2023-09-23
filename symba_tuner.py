import torch
import optuna
from engine import Trainer, Config

def objective(trail):
    
    optimizer_lr = trail.suggest_categorical("optimizer_lr", [0.0001, 0.0005, 0.001])
    optimizer_weight_decay = trail.suggest_categorical("optimizer_weight_decay", [0.00001, 0.00005, 0.0001])
    dropout = trail.suggest_categorical("dropout", [0.1, 0.3, 0.5])
    label_smoothing = trail.suggest_categorical("label_smoothing", [0, 0.001, 0.01])
    vocab_size = trail.suggest_categorical("vocab_size", [2264, 2875])
    embedding_size = trail.suggest_categorical("embedding_size", [512, 256, 1024])
    hidden_dim = trail.suggest_categorical("hidden_dim", [512, 256, 8192])
    clip_grad_norm = trail.suggest_categorical("clip_grad_norm", [-1, 1, 5])
    
    config = Config(
        experiment_name = f"hp_tuning/lr_{optimizer_lr}_wd_{optimizer_weight_decay}_gn_{clip_grad_norm}_dp_{dropout}_ls_{label_smoothing}_vs_{vocab_size}_es_{embedding_size}_hs_{hidden_dim}",
        root_dir = "./",
        device = "cuda",
        model_name = "seq2seq_transformer",
        dataset_name = "QED_Amplitude",
        epochs=10,
        seed=42,
        training_batch_size = 64,
        test_batch_size = 128,
        optimizer_type = "adam",
        optimizer_lr = optimizer_lr,
        optimizer_weight_decay = optimizer_weight_decay,
        clip_grad_norm = clip_grad_norm,
        scheduler_type = "none",
        vocab_size = vocab_size,
        embedding_size = embedding_size,
        hidden_dim = hidden_dim,
        num_encoder_layers = 3,
        num_decoder_layers = 3,
        nhead = 8,
        dropout = dropout
    )
    
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing = label_smoothing)
    
    class SymbaTrainer(Trainer):
        
        def criterion(self, y_pred, y_true):
            return loss_fn(y_pred, y_true)
        
        def on_eval_end(self, valid_accuracy, valid_loss):
            if self.best_accuracy<0.92 and self.current_epoch>5:
                self.stop_training = True
        
    trainer = SymbaTrainer(config)
    trainer.fit()
    
    return trainer.best_val_loss
    
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=4000)
    
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
