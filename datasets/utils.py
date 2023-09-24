import random
import os
import re
import pandas as pd
import csv
from transformers import AutoTokenizer

def preprocess(data, is_square=False, is_fynman=False):
    '''Used to preprocess the amplitude and squared amplitude text'''
    
    if is_square:
        for r in (('*', '*'), (',', ' , '), ('*(', ' *( ') , ('([', '[ '), ('])', ' ]'), ('[', '[ '), 
                  (']', ' ]'), ('[ start ]', '[start]'), ('[ end ]', '[end]'), (' - ', ' -'), 
                  (' + ',' +' ) ,('/', ' / ') ,('  ', ' ')) :
            data = data.replace(*r) 
        data = re.sub(r"\*(s_\d+\*s_\d+)", r"* \1", data)
        data = re.sub(r"\*(s_\d+\^\d+\*s_\d+)", r"* \1", data)
        data = re.sub(r"\*(m_\w+\^\d+\*s_\d+)", r"* \1", data)
        data = re.sub(r"(m_\w+\^\d+)", r" \1 ", data)
        data = data.replace('  ', ' ')
        
        return data
    
    elif is_fynman:
        for r in (('(', '('), (')', ')'), ('  ', ' '), (' e(', 'e(m_e,-1,' ),(' mu(', 'mu(m_mu,-1,'), 
                  (' u(', ' u(m_u,2/3,'), (' d(', 'd(m_d,-1/3,'), (' t(', ' t(m_t,-1,') ,(' s(', 's(m_s,-1/3,'),
                  (' tt(', ' tt(m_tt,-1,'), (' c(', 'c(m_c,2/3,'),(' b(', 'b(m_b,-1/3,'), ('Anti ', 'Anti,'), 
                  ('Off ', 'Off,'), ('  ', ' ')): 
            data = data.replace(*r) 
        
        return data

    else:
        for r in (('}', '}'),('{', ' {'), (' + ',' +' ), (' - ', ' -') ,('*', '* '), ('(* )', '(*)'),
                  ('^', '^') , ('(', ' ('),(')', ')'),('/', ' /')  ,('  ', ' ') ) : 
            data = data.replace(*r) 
        
        return data

def max_len(data):
    '''return len of the data'''
    
    l = len(data[data.index(max(data, key=len))].split())
    return l

def create_csv_json(path):
    '''This function is used to create the csv and json file of amplitude/fynman_diagram and square amplitude'''
    
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        
    dataset_name = path.split('/')[1]
    text_pairs = []
    for line in lines[: min(len(lines), len(lines)-1)]:
        if "Amplitude" in dataset_name:
            intr, amp, sqamp, t = line.split(':')
        else:
            intr, amp, sqamp, t = line.split('>')
        text_pairs.append((amp, sqamp))

    #Removing long amplitudes/squared amplitudes
    text_pairs1 = []
    for i in range(len(text_pairs)):
        if "Amplitude" in dataset_name:
            if len(text_pairs[i][0]) < 2000  and len(text_pairs[i][1]) < 1800:
                text_pairs1.append(text_pairs[i])
        else:
            if "QED" in dataset_name:
                if  len(text_pairs[i][1]) < 1800:
                    text_pairs1.append(text_pairs[i])
            else:
                if  len(text_pairs[i][1]) < 800:
                    text_pairs1.append(text_pairs[i])
            
    text_pairs = text_pairs1
    
    processed_text_pairs = []
    for i in range(len(text_pairs)):
        if "Amplitude" in dataset_name:
            processed_text_pairs.append((preprocess(text_pairs[i][0]), preprocess(text_pairs[i][1], True)))
        else:
            processed_text_pairs.append((preprocess(text_pairs[i][0], False, True), preprocess(text_pairs[i][1], True)))

    text_pairs = processed_text_pairs
    
    # Splitting the dataset
    num_val_samples = int(0.15 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    val_pairs  = text_pairs[num_train_samples : num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples :]

    train_input_texts = [pair[0] for pair in train_pairs] # Amplitudes
    train_output_texts = [pair[1] for pair in train_pairs] # Squared Amplitudes
    
    val_input_texts = [pair[0] for pair in val_pairs] # Amplitudes
    val_output_texts = [pair[1] for pair in val_pairs] # Squared Amplitudes
    
    test_input_texts = [pair[0] for pair in test_pairs] # Amplitudes
    test_output_texts = [pair[1] for pair in test_pairs] # Squared Amplitudes
    
    if "Amplitude" in dataset_name:
        key_0 = "Amplitude"
    else:
        key_0 = "Feynman_Diagram"
    
    print("===> Creating CSV and JSON file.\n")
    
    raw_data_train = {key_0: train_input_texts,
                      'Squared_Amplitude': train_output_texts}
    df_train = pd.DataFrame(raw_data_train, columns=[key_0, 'Squared_Amplitude'])
    
    raw_data_val = {key_0: val_input_texts,
                      'Squared_Amplitude': val_output_texts}
    df_val = pd.DataFrame(raw_data_val, columns=[key_0, 'Squared_Amplitude'])
    
    raw_data_test = {key_0: test_input_texts,
                      'Squared_Amplitude': test_output_texts}
    df_test = pd.DataFrame(raw_data_test, columns=[key_0, 'Squared_Amplitude'])
    
    df_train.to_csv('./data/'+dataset_name+'/train.csv', index=False)
    df_val.to_csv('./data/'+dataset_name+'/val.csv', index=False)
    df_test.to_csv('./data/'+dataset_name+'/test.csv', index=False)

    df_train.to_json('./data/'+dataset_name+'/train.json', orient='records', lines=True)
    df_val.to_json('./data/'+dataset_name+'/val.json', orient='records', lines=True)
    df_test.to_json('./data/'+dataset_name+'/test.json', orient='records', lines=True)


    print("==> CSV and JSON files created.\n")
    
# Train and save config for BART and LED model
def create_vocab(config): 
    amplitude_expressions = []
    squared_amplitude_expressions = []
    
    path = "./data/"+config.dataset_name+'/train.csv'
    
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == "Amplitude":
                continue
            if len(row) == 2:
                amplitude_expressions.append(row[0])
                squared_amplitude_expressions.append(row[1])
                
    all_expressions = amplitude_expressions + squared_amplitude_expressions
    
    model_name = config.model_name
    help_dict = {'bart-base':'facebook/bart-base', 'bart-large':'facebook/bart-large', 'LED-base':'allenai/led-base-16384'}
    old_tokenizer = AutoTokenizer.from_pretrained(help_dict[model_name])
    
    print(f"==> Training the tokenizer on {config.dataset_name}\n"}
    tokenizer = old_tokenizer.train_new_from_iterator(all_expressions, config.vocab_size)
    tokenizer.save_pretrained("./data/"+config.dataset_name+f"/{model_name}_tokenizer")
    print("==> New Tokenizer Created\n"}
    
