''' This module help us to plot the training loss, validation loss, validation accuracy
and test accuracy '''
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Plotter:
    def __init__(self, config):
        self.config = config
        self.epochs = self.config.epochs
        self.path = './'+self.config.model_name+'/'+self.config.dataset_name+'/'+self.config.experiment_name+'/checkpoint.pth'
        with open('./'+self.config.model_name+'/'+self.config.dataset_name+'/'+self.config.experiment_name+'/score.txt') as f:
            self.test_score = f.read()
        self.test_token_score = float(self.test_score.split('\n')[0].split(" ")[3])
        self.test_seq_score = float(self.test_score.split('\n')[1].split(" ")[3])
        self.state = torch.load(self.path)
        self.train_loss_list = self.state['train_loss_list']
        self.valid_loss_list = self.state['valid_loss_list']
        self.valid_accuracy_list = self.state['valid_accuracy_tok_list']
        
    def plot(self):
        X = [i for i in range(self.epochs)]
        plt.plot(X, self.train_loss_list, label="training loss", color="#FFA07A")
        plt.plot(X, self.valid_loss_list, label="validation loss", color="#4B0082")
        plt.plot(X, self.valid_accuracy_list, label="validation accuracy", color="#ADFF2F")
        plt.scatter(self.epochs-1, float(self.test_token_score), marker="*", label="test accuracy (Token)")
        plt.scatter(self.epochs-1, float(self.test_seq_score), marker="x", label="test accuracy (Sequence)")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss/Accuracy")
        plt.xticks(np.arange(0, self.epochs, 1))
        
        plt.legend(loc="lower left", framealpha=0.69)
        plt.show()
        
