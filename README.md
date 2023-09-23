# Symba_torch

This is the repository for my Google Summer of Code Project [Symbolic empirical representation of squared amplitudes in high-energy physics](https://summerofcode.withgoogle.com/programs/2023/projects/DLza6brS) 

### Overview
In particle physics, a cross section is a measure of the likelihood that particles will interact or scatter with one another when they collide. It is a fundamental quantity that is used to describe the probability of certain interaction occurring between particles. The determination of cross-sectional area necessitates the computation of the squared amplitude, as well as the averaging and summation over the internal degrees of freedom of the particles involved. This project aims to apply symbolic deep learning techniques to predict the squared amplitudes and cross section for high energy physics.

### Installation 
Install required packages:
```python
pip install torch
pip install timm
pip install transformers
```
After installing the required packages, clone this repository:
```bash
git clone https://github.com/neerajanand321/Symba_torch.git
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
Refer to the [symba_example.ipynb](https://github.com/neerajanand321/Symba_torch/blob/main/symba_example.ipynb).
