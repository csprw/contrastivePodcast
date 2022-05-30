"""
Index values for elasticsearch.
Author: Casper Wortmann
Usage: 
    start elasticsearch (./bin/elasticsearch)
    Load searh results in http://localhost:9200/_search
"""
import h5py
import json, csv
import os
from pathlib import Path
import math
import pandas as pd
import numpy as np

import torch 
from transformers import AutoTokenizer
from elasticsearch_dsl import Document, Integer, Text, DenseVector
from elasticsearch_dsl.connections import connections

import time
import requests
from tqdm import tqdm
from dataclasses import dataclass

from dataclasses_json import dataclass_json
from dacite import from_dict
from nltk import tokenize
from omegaconf import OmegaConf 
from pprint import pprint

from transformers import DistilBertTokenizer
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

import src.data
from src.data import load_metadata, find_paths, relative_file_path

import sys
sys.path.append('../contrastive_mm2/') 
from gru_test2 import mmModule, Cfg
from gru_test2 import MMloader


from prepare_index_sentencelevel2 import read_metadata_subset

# Load static configuration variables. 
config_path = "./config.yaml"
conf = OmegaConf.load(config_path)
print("[cudacheck] Is cuda available? ", torch.cuda.is_available())

 
if __name__ == "__main__":
    print("[main]")
    model_path = "E:\msc_thesis\code\contrastive_mm2\logs\lisa_v2-simcse_loss_rnn_relu_768_2e-05_2022-05-17_06-58-44"
    model_path = '/Users/casper/Documents/UvAmaster/b23456_thesis/msc_thesis/code/contrastive_mm2/logs/windows_gru2-clip_loss_gru_gelu_768_5e-05_2022-05-26_21-51-25'
    model_path = '/Users/casper/Documents/UvAmaster/b23456_thesis/msc_thesis/code/contrastive_mm2/logs/windows_gru2-clip_loss_followup'

    transcripts_path = '/Users/casper/Documents/UvAmaster/b23456_thesis/msc_thesis/code/data/sp/podcasts-no-audio-13GB/podcasts-transcripts'
    # transcripts_path = 'E:/msc_thesis/code/data/sp/podcasts-no-audio-13GB/podcasts-transcripts'

    # Whether we create for the train or the tes split. 
    traintest = 'test'


    # Creating output directory.
    save_name = os.path.split(model_path)[-1]
    save_path = os.path.join(conf.yamnet_topic_embed_path, save_name + "_" + traintest)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Reading the metadata.
    metadata_testset, topics_df, topics_df_targets = read_metadata_subset(conf, traintest='test')

    # Remove duplicate rows
    metadata_testset = metadata_testset.drop_duplicates(subset=['episode_filename_prefix']).sort_values('episode_filename_prefix')
    print("[main] Topic test set loaded: ", len(metadata_testset))

    # Loading the model.
    print("[Load model] from ", model_path)
    model_weights = os.path.join(model_path, "output/full_model_weights.pth")
    print("[Load model] weights loaded")
    model_config_path = os.path.join(model_path, 'config.json')

    # Opening JSON file
    f = open(model_config_path)
    model_config = json.load(f)
    print("using config: , ", model_config)
    print("\n\n")
    CFG = from_dict(data_class=Cfg, data=model_config)
    CFG.device = "cuda" if torch.cuda.is_available() else "cpu"
    # mutual_embed_dim = CFG.final_projection_dim
    print("[Load model] config loaded: ", CFG)

    # Load model and tokenizer
    full_model = mmModule(CFG)
    full_model.load_state_dict(torch.load(model_weights,  map_location=CFG.device))                      
    full_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(CFG.text_model_name)


    # Create dataloader of test set
    data_loader = MMloader(CFG)
    # train_loader = data_loader.train_loader
    # val_loader = data_loader.val_loader
    test_loader = data_loader.test_loader # Not used!

    # TODO: move this into __init__ of dataloader
    data_loader.test_dataset.tokenizer = tokenizer
    data_loader.test_dataset.text_max_length = CFG.text_max_length


    do_vali = True

    if do_vali:
        iterator = iter(test_loader)
        max_steps = len(test_loader) 
        print("START TRAIN FOOR STEPS: ",  max_steps)


        for step in range(max_steps):
            print("step {}/{}".format(step, max_steps))
            batch = next(iterator)
            
            (tok_sentences, audio_features, seq_len, targets) = batch

            with torch.no_grad():
                reps_sentences = full_model.text_model(tok_sentences)['sentence_embedding']
                reps_audio = full_model.audio_model((audio_features, seq_len))
                
                audio_logits =  (reps_audio @ reps_sentences.t()) * CFG.scale
                text_logits = audio_logits.t()
                
                audio_logits = audio_logits / audio_logits.norm(dim=1, keepdim=True)
                text_logits = text_logits / text_logits.norm(dim=1, keepdim=True)
                
                probs = audio_logits.softmax(dim=-1).cpu().numpy()

                probs = audio_logits.softmax(dim=-1).cpu().numpy()
                ground_truth = torch.arange(128)
                acc = torch.eq(torch.tensor(probs.argmax(axis=0)), ground_truth).sum() / ground_truth.shape[0]
                print(ground_truth.shape[0])
                print("accuracy", acc.item())







        