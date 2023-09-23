# SYMBA_Pytorch

This is the repository for my Google Summer of Code Project [Symbolic empirical representation of squared amplitudes in high-energy physics](https://summerofcode.withgoogle.com/programs/2023/projects/DLza6brS) 

### Overview
In particle physics, a cross section is a measure of the likelihood that particles will interact or scatter with one another when they collide. It is a fundamental quantity that is used to describe the probability of certain interaction occurring between particles. The determination of cross-sectional area necessitates the computation of the squared amplitude, as well as the averaging and summation over the internal degrees of freedom of the particles involved. This project aims to apply symbolic deep learning techniques to predict the squared amplitudes and cross section for high energy physics.

### Code Structure
```python
SYMBA_Pytorch
|__ datasets
|          |__ Data.py # Code for dataset cleaning, tokenization etc.
|          |__ __init__.py
|          |__ registry.py # All the dataset must be registered
|          |__ utils.py # Helper modules
|__ engine
|         |__ __init__.py
|         |__ config.py # All the required configuration for training of models.
|         |__ plotter.py # Used for plotting the loss and accuracy
|         |__ predictor.py # Used for prediction 
|         |__ trainer.py # Used for training of models.
|         |__ utils.py # Helper modules
|__ models
|         |__ BART.py # Code for BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
|         |__ LED.py # Code for Longformer Encoder Decoder 
|         |__ __init__.py
|         |__ registery.py # All the models must be registered 
|         |__ seq2seq_transformer.py # Code for Sequence to sequence Transformer
|__ runs
|       |__ bart-base_trainer.sh # Script to run bart-base model using terminal
|       |__ seq2seq_trainer.sh # Script to run sequence to sequence transformer using terminal
|__ symba_trainer.py # Used inside bash script for training.
|__ symba_tuner.py # Used for hyperparameter optimization using Optuna
|__ symba_example.ipynb # Example Notebook.
```

### Installation 
Install required packages:
```python
pip install torch
pip install timm
pip install transformers
```
After installing the required packages, clone this repository:
```bash
git clone https://github.com/ML4SCI/SYMBA_Pytorch.git
```
### Training 
Training can be done either using terminal or notebook.
#### Using Terminal
Use the following command to train the network on single GPU:
```bash
python symba_trainer.py --experiment_name="Example" --model_name="seq2seq_transformer" \
                        --dataset_name="QCD_Amplitude" --epochs=30 --learning_rate=0.0001 
```
Use the following command to train the network on multiple GPU:
```bash
torchrun symba_trainer.py --experiment_name="Example" --model_name="seq2seq_transformer" \
                          --dataset_name="QCD_Amplitude" --distributed=True --epochs=30 --learning_rate=0.0001 
```
#### Using Notebook
Refer to the [symba_example.ipynb](https://github.com/ML4SCI/SYMBA_Pytorch/blob/main/symba_example.ipynb).
