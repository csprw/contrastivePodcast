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
import logging
import sys
from argparse import ArgumentParser
import time
from time import time
from datetime import datetime
from tkinter import E
import h5py
import json, csv
import os
from pathlib import Path
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter, OrderedDict
from pprint import pprint
from dacite import from_dict

from typing import List, Dict, Optional, Union, Tuple, Iterable, Type, Callable
from dataclasses import dataclass, asdict
import torch 
from transformers import AutoTokenizer
from elasticsearch_dsl import Document, Integer, Text, DenseVector
from elasticsearch_dsl.connections import connections

import torch 
from torch.utils import data as datautil
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AdamW, AutoConfig, T5Config
from transformers import get_constant_schedule, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

from torch import nn, Tensor
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

import psutil, gc

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
from pathlib import Path
import sys
sys.path.append('../contrastive_mm2/') 
# from gru_test2 import mmModule, Cfg
# from gru_test2 import MMloader


from prepare_index_sentencelevel2 import read_metadata_subset

# Load static configuration variables. 
config_path = "./config.yaml"
conf = OmegaConf.load(config_path)
print("[cudacheck] Is cuda available? ", torch.cuda.is_available())

from torch import topk
from sklearn.metrics import ndcg_score

def precision_at_k(predicted, k):
    if k == 0:
        print("precision from k ===0?")
        raise
    pred = (predicted[:k])
    relevant = np.sum(pred)
    num_recommended = float(k)
    p_at_k = relevant / num_recommended
    return p_at_k

# Recall@k = (# of recommended items @k that are relevant) / (total # of relevant items)
def recall_at_k(predicted, num_relevant, k):
    if num_relevant == 0:
        return None
    pred = (predicted[:k])
    relevant = np.sum(pred)
    r_at_k = relevant / num_relevant

    return r_at_k

########################### DELETE 

            
@dataclass
class Cfg:
    batch_size: int = 128
    num_epochs: int = 1
    loss_type: str = 'simcse_loss'
    lr: float = 5e-5
    # device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # For the audio module
    audio_encoder_input: int = 1024
    audio_hidden_dim: int = 768
    audio_layer_dim: int = 2
    audio_activation: str = 'relu'

    # For the text module
    text_model_name : str = 'distilbert-base-uncased'
    text_tokenizer: str = "distilbert-base-uncased"
    text_pooling: str = 'mean'   #['cls', 'mean']
    text_max_length: int = 32

    # For the projection_head modules
    text_proj_head: str = 'None'
    audio_proj_head: str = 'None'
    text_activation: str = 'gelu'
    mutual_embedding_dim: int = 768


    final_projection_dim: int = 768  # [256 or 768]
    audio_dropout: float = 0.1
    text_dropout: float = 0.1
    weight_decay: float = 0.01

    text_activation: str = ''
    audio_activation: str = ''
    scale_type: str = ''
    scale: int = 20
    pad_pack: bool = False
    normalize: bool = False
    eval_every: int = 1
    print_every: int = 1
    log_name: str = 'logname'

    train_dataset: str = ''
    val_dataset: str = ''
    test_dataset: str = ''
    seed: int = 100

    save_model: bool = False
    save_checkpoint: bool = False
    load_model_path: str = ''
    load_model: bool = False
    load_checkpoint: bool = False
    load_checkpoint_path: str = ''


class MMloader(object):
    """
    Loads one of the supported datasets.
    Supported datasets:
        sts dataset
        sample data (scraped)
        SP dataset (future)
    """
    def __init__(self, CFG, kwargs={}):

        train_dataset_name = CFG.train_dataset
        val_dataset_name = CFG.val_dataset
        test_dataset_name = CFG.test_dataset

        batch_size = CFG.batch_size
        self.batch_size = batch_size
        self.device = CFG.device

        if CFG.audio_proj_head in ['gru', 'rnn']:
            self.load_full = True
        else:
            self.load_full = False

        print("[MMloader] train dataset ", train_dataset_name)

        # Get the datasets
        if train_dataset_name == "sp_sample":
            train_dataset = self.get_sp_dataset(directory=conf.sp_sample_path, traintest="train", load_full=self.load_full, device=self.device)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)  ### TODO: SHUFFLE=TRUE [DEL]
        elif train_dataset_name == "sp":
            train_dataset = self.get_sp_dataset(directory=conf.sp_path,  traintest="train", load_full=self.load_full, device=self.device)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        else:
            raise Exception('Unknown dataset')
        
        self.train_dataset = train_dataset
        self.train_loader.collate_fn = self.train_dataset.collate_fn
        print("[MMloader] train dataset loaded, length: ", len(self.train_dataset))

        if val_dataset_name == "sp_sample":
            val_dataset = self.get_sp_dataset(directory=conf.sp_sample_path,  traintest="val", load_full=self.load_full, device=self.device)
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs) 
        elif val_dataset_name == "sp":
            val_dataset = self.get_sp_dataset(directory=conf.sp_path,  traintest="val",  load_full=self.load_full, device=self.device)
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        else:
            raise Exception('Unknown dataset')
        self.val_dataset = val_dataset
        self.val_loader.collate_fn = self.val_dataset.collate_fn

        if test_dataset_name == "sp_sample":
            test_dataset = self.get_sp_dataset(directory=conf.sp_sample_path,  traintest="test", load_full=self.load_full, device=self.device)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs) 
        elif test_dataset_name == "sp":
            test_dataset = self.get_sp_dataset(directory=conf.sp_path,  traintest="test",  load_full=self.load_full, device=self.device)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        else:
            raise Exception('Unknown dataset')
        self.test_dataset = test_dataset
        self.test_loader.collate_fn = self.test_dataset.collate_fn
        print("[MMloader] test dataset loaded, length: ", len(test_dataset))

    def get_sp_dataset(self, directory=conf.sp_sample_path, traintest="train", load_full=False, device=None):
        dataset =  spDatasetNoMemory(directory=directory, traintest=traintest,  load_full=load_full, device=device)
        return dataset

class spDatasetNoMemory(datautil.Dataset):
    """
    Spotify Podcast dataset dataloader. 
    """
    def __init__(self,  directory=conf.sp_sample_path, traintest="train", load_full=False, device=None):
        print("[spDataset] init from directory ", directory, traintest)
        directory = os.path.join(directory, traintest)
        h5py_files = list(Path(directory).glob('*.h5'))
        print("[spDataset] found {} h5py files".format(len(h5py_files)))
        self.max_embed_dim = 0
        self.device = device
        idx2file = {}
        self.h5py_idx2file = h5py_files
        sample_idx = 0

        self.load_full = load_full

        for h5idx, h5py_file in enumerate(h5py_files):    
            print("[del] h5py_file: ", h5py_file)
            f = h5py.File(h5py_file, 'r')

            for sentidx in range(len(f['sentences'])):
                idx2file[sample_idx] = (h5idx, sentidx)
                sample_idx += 1
 
            self.max_embed_dim = max(self.max_embed_dim, f.attrs['max_embed_dim'])
            f.close()
            if h5idx % 10 == 0 and h5idx > 0:
                print("[spdataset] loaded {}/{}".format(h5idx, len(h5py_files)))
        self.idx2file = idx2file


        if traintest == 'test':
            self.return_targets = True
        else: 
            self.return_targets = False

        if self.load_full:
            self.collate_fn = self.full_batching_collate
        else:
            self.collate_fn = self.mean_batching_collate

    def __len__(self):
        """ Denotes the total number of utterances """
        return len(self.idx2file.keys())

    def __getitem__(self, index):
        """ Return one item from the df """

        if self.return_targets:
            h5py_idx, sent_idx = self.idx2file[index]
            h5py_file = self.h5py_idx2file[h5py_idx]
            f = h5py.File(h5py_file, 'r')

            # print("[del] Return targets")
            sent = f['sentences'][sent_idx]
            full_embeds = torch.Tensor(np.array(f[str(sent_idx)]))

            # print("1: ", str(f['filenames'][sent_idx]))
            # print("2: ", (f['segment_ts'][sent_idx]))
            # target = f['filenames'][sent_idx].decode("utf-8") + '_' + str(f['segment_ts'][sent_idx])
            target = f['seg_ts'][sent_idx].decode("utf-8") 

            # print("Target: ", target)

            sample = (sent, full_embeds, target)


        elif self.load_full:
            h5py_idx, sent_idx = self.idx2file[index]
            h5py_file = self.h5py_idx2file[h5py_idx]

            f = h5py.File(h5py_file, 'r')

            # TODO: FIND OUT WHY IT SOMETIMES IS LIKE THIS:
            # sent = f['sentences'][sent_idx].decode("utf-8")
            sent = f['sentences'][sent_idx]
            full_embeds = torch.Tensor(np.array(f[str(sent_idx)]))
            sample = (sent, full_embeds)
            f.close()
            
        else:
            h5py_idx, sent_idx = self.idx2file[index]
            h5py_file = self.h5py_idx2file[h5py_idx]

            f = h5py.File(h5py_file, 'r')
            
            # TODO: FIND OUT WHY IT SOMETIMES IS LIKE THIS:
            # sent = f['sentences'][sent_idx].decode("utf-8")
            sent = f['sentences'][sent_idx]

            mean_embeds = torch.Tensor(f['mean_embeddings'][sent_idx])
            sample = (sent, mean_embeds)
            f.close()
            
        return sample

    def full_batching_collate(self, batch):
        """ Return a batch """
        text_embeds = []
        audio_embeds = []
        lengths  = []
        
        for example in batch:
            text_embeds.append(example[0])
            audio_embeds.append(example[1])
            lengths.append(len(example[1]))
        
        # Pad the audio embeddings
        padded_audio_embeds = pad_sequence(audio_embeds, batch_first=True).to(self.device)
        # print("[del] paded device: ", padded_audio_embeds.is_cuda)
        
        # Tokenize text
        text_embeds = self.tokenizer(
            text_embeds, padding=True, truncation=True, max_length=self.text_max_length, return_tensors='pt'
        ).to(self.device)
        # print("[del] text_embeds device: ", text_embeds.is?_cuda)

        if self.return_targets:
            # print("[del] RETURN TARGS")
            targs = []
            for example in batch:
                targs.append(example[2])
            return text_embeds, padded_audio_embeds, lengths, targs
        else:
            # print("[del] not return targs")
            return text_embeds, padded_audio_embeds.float(), lengths

    def mean_batching_collate(self, batch):
        """ Return a batch """
        text_embeds = []
        audio_embeds = []
        lengths  = []
        for example in batch:
            text_embeds.append(example[0])
            audio_embeds.append(example[1])

        # Combine audio embeddings to a Tensor
        audio_embeds = torch.stack(audio_embeds).to(self.device)

        # Tokenize text
        max_length = 32 # TODO: is dit nodig?
        text_embeds = self.tokenizer(
            text_embeds, padding=True, truncation=True, max_length=max_length, return_tensors='pt'
        ).to(self.device)
        
        return text_embeds, audio_embeds, lengths

    

################################################################################
# Textmodules
class TextEncoder(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        model_name_or_path = CFG.text_model_name
        self.config_keys = ['max_seq_length', 'do_lower_case']
        

        # model_args: Dict = {}
        self.do_lower_case = None
        # my_dropout = 0.1
        # cache_dir = None
        # config_kwargs = {
        #     "attention_dropout": my_dropout,
        #     "dropout": my_dropout,
        # }

        config = AutoConfig.from_pretrained(model_name_or_path)
        self.hidden_dim = config.dim
        print("[Transformer_multimodal] Configurations used: \n", config)
        
        # This is self._load_model in mm2

        self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.max_seq_length = CFG.text_max_length

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0
        self.pooling = CFG.text_pooling # TODO: delete this line?

    def forward(self, text_embeds):
        trans_features = {'input_ids': text_embeds['input_ids'], 'attention_mask': text_embeds['attention_mask']}
        #trans_features = {'input_ids': text_embeds['input_ids'].to('cuda'), 'attention_mask': text_embeds['attention_mask'].to('cuda')}
        
        # print(trans_features['input_ids'].is_cuda, trans_features['attention_mask'].is_cuda)
        # print(trans_features.is_cuda)
        output = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output[0]
        features = {'input_ids':text_embeds['input_ids'], 'attention_mask':text_embeds['attention_mask'],'token_embeddings': output_tokens}
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size


class Pooling(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.

    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode: Can be a string: mean/max/cls. If set, overwrites the other pooling_mode_* settings
    :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
    :param pooling_mode_max_tokens: Use max in each dimension over all tokens.
    :param pooling_mode_mean_tokens: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but devide by sqrt(input_length).
    """
    def __init__(self,
                 word_embedding_dimension: int = 768,
                 pooling_mode: str = None,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 ):
        super(Pooling, self).__init__()

        self.config_keys = ['word_embedding_dimension',  'pooling_mode_cls_token', 'pooling_mode_mean_tokens', 'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens']

        if pooling_mode is not None:        #Set pooling mode by string
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ['mean', 'max', 'cls']
            pooling_mode_cls_token = (pooling_mode == 'cls')
            pooling_mode_max_tokens = (pooling_mode == 'max')
            pooling_mode_mean_tokens = (pooling_mode == 'mean')

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens

        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens, pooling_mode_mean_sqrt_len_tokens])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def __repr__(self):
        return "Pooling({})".format(self.get_config_dict())

    def get_pooling_mode_str(self) -> str:
        """
        Returns the pooling mode as string
        """
        modes = []
        if self.pooling_mode_cls_token:
            modes.append('cls')
        if self.pooling_mode_mean_tokens:
            modes.append('mean')
        if self.pooling_mode_max_tokens:
            modes.append('max')
        if self.pooling_mode_mean_sqrt_len_tokens:
            modes.append('mean_sqrt_len_tokens')

        return "+".join(modes)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            cls_token = features.get('cls_token_embeddings', token_embeddings[:, 0])  # Take first token by default
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)
        return Pooling(**config)

################################################################################
# Audiomodules
class SequentialAudioModel(nn.Module):
    def __init__(self, CFG):
        super(SequentialAudioModel, self).__init__()
        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = CFG.audio_hidden_dim
        self.layer_dim = CFG.audio_layer_dim
        self.device = CFG.device
        self.audio_model = CFG.audio_proj_head

        # RNN layers
        if self.audio_model == 'rnn':
            self.seq_model = nn.RNN(CFG.audio_encoder_input, 
                    CFG.audio_hidden_dim, CFG.audio_layer_dim, 
                    batch_first=True, dropout=CFG.audio_dropout)
        elif self.audio_model == 'gru':
            self.seq_model = nn.GRU(
                    input_size=CFG.audio_encoder_input, 
                    hidden_size=CFG.audio_hidden_dim, num_layers=CFG.audio_layer_dim, 
                    batch_first=True, dropout=CFG.audio_dropout)

        # Fully connected layer
        self.fc = nn.Linear(CFG.audio_hidden_dim, CFG.mutual_embedding_dim)
        self.softmax = nn.Softmax(dim=1)

        # pad_pack = args.pad_pack
        self.pad_pack = CFG.pad_pack
        self.use_softmax = True
        # TODO: print("[TODO] pad_pack is set to false")
        # To CG

    def forward(self, audio_seq):
        features, length = audio_seq

        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, features.size(0), self.hidden_dim).requires_grad_().to(self.device)

        if length != None:
            # print("lengths: ", length)
            # print("should be [BS* SL * ?] ", x.shape)
            if self.pad_pack:
                # print("!! PACK PAD ON!!")
                # Pack the features such that we do not compute zero products
                features = pack_padded_sequence(features, length, batch_first=True, enforce_sorted=False)

            #embedded = torch.nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
            #packed_output, h0 = self.rnn(embedded, h0.detach())

            out, h0 = self.seq_model(features, h0.detach())
            if self.pad_pack:
                out, output_lengths = pad_packed_sequence(out, batch_first=True)
            
            do_trick = True
            # do_trick = False

        else:
            # Forward propagation by passing in the input and hidden state into the model
            #out, h0 = self.rnn(x, h0.detach())
            out, h0 = self.seq_model(features, h0.detach())
            # print("[rnn] out forward: [bs, sl, hidden]", out.shape)
            do_trick=False

        # Convert the final state to our desired output shape (batch_size, output_dim)
        if do_trick:
            out = h0[-1, :, :]
        else:
            # Reshaping the outputs
            # so that it can fit into the fully connected layer
            out = out[:, -1, :]
            # print("trick outshape:[bs * hidden] ", out.shape)

        out = self.fc(out)
        if self.use_softmax:
            out = self.softmax(out)
        # print("[rnn] ifnal out: [bs*output]", out.shape)
        return out

class simple_ProjectionHead(nn.Module):
    """
    :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
    """
    def __init__(self, CFG):
        super(simple_ProjectionHead, self).__init__()

        # OG input: input_dim, hidden_dim=768, output_dim=768, proj_type = 'gelu'
        
        self.input_dim = CFG.audio_encoder_input
        self.hidden_dim = CFG.audio_hidden_dim
        self.output_dim = CFG.final_projection_dim
        self.activation  = CFG.audio_activation
        self.config_keys = ['input_dim', 'hidden_dim', 'output_dim', 'activation']
        dropout = CFG.dropout

        if self.activation == 'relu':
            self.simple_model = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.BatchNorm1d(num_features=self.hidden_dim), # TODO: experiment with batchnormalization
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        else:
            self.simple_model = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(self.output_dim),
            )

    def __repr__(self):
        return "Simple_projection_head({}) ".format(self.get_config_dict())

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def forward(self, feat_len):
        """Returns token_embeddings, cls_token"""

        features, _ = feat_len
        features = self.simple_model(features)

        return features

class text_ProjectionHead(nn.Module):
    # TODO: This module is depricated.
    def __init__(self, CFG):

        super().__init__()
        embedding_dim=CFG.mutual_embedding_dim
        projection_dim=CFG.final_projection_dim
        dropout=CFG.text_dropout

        self.activation  = CFG.text_activation

        if self.activation == 'relu':
            self.net = nn.Sequential(
                nn.Linear(embedding_dim, projection_dim),
                nn.BatchNorm1d(num_features=projection_dim), # TODO: experiment with batchnormalization
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim),
            )

        else:
            self.net = nn.Sequential(
                nn.Linear(embedding_dim, projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(projection_dim),
            )
    
    def forward(self, x):
        x = self.net(x)
        return x

class mmModule(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        # self.temperature=CFG.temperature
        self.batch_size = CFG.batch_size
        self.device = CFG.device

        self.loss_type = CFG.loss_type

        if CFG.text_proj_head == 'simple_projection_head':
            self.text_encoder = TextEncoder(CFG)
            self.text_projection = text_ProjectionHead(CFG)

            text_modules = [self.text_encoder, self.text_projection]
        elif CFG.text_proj_head.lower() == 'none':
            text_encoder = TextEncoder(CFG)
            # pooling_mode_cls_token: bool = False,
            #  pooling_mode_max_tokens: bool = False,
            #  pooling_mode_mean_tokens: bool = True,
            #  pooling_mode_mean_sqrt_len_tokens: bool = False,
            if CFG.text_pooling == 'mean':
                pooling_model = Pooling(text_encoder.get_word_embedding_dimension(),
                pooling_mode_mean_tokens = True)
            elif CFG.text_pooling == 'cls':
                pooling_model = Pooling(text_encoder.get_word_embedding_dimension(),
                pooling_mode_mean_tokens = False,
                pooling_mode_cls_token = True)

            pooling_model = Pooling(text_encoder.get_word_embedding_dimension())
            text_modules = [text_encoder, pooling_model]

        if CFG.audio_proj_head in ['rnn', 'gru']:
            audio_encoder = SequentialAudioModel(CFG)
            audio_modules = [audio_encoder]
        elif CFG.audio_proj_head in ['simple_projection_head']:
            audio_encoder = simple_ProjectionHead(CFG)
            audio_modules = [audio_encoder]
        
        # Create the full model
        if text_modules is not None and not isinstance(text_modules, OrderedDict):
            text_modules = OrderedDict([(str(idx), module) for idx, module in enumerate(text_modules)])
        if audio_modules is not None and not isinstance(audio_modules, OrderedDict):
            audio_modules = OrderedDict([(str(idx), module) for idx, module in enumerate(audio_modules)])

        self.text_model = nn.Sequential(text_modules)
        self.audio_model = nn.Sequential(audio_modules)

        # self.temperature = CFG.temperature
        self.eval_every = CFG.eval_every
        self.print_every = CFG.print_every
        self.batch_size = CFG.batch_size
        self.device = CFG.device

        self.log_name = CFG.log_name
        self.init_logging()

        self.best_loss = float('inf')
        # self.loss1 = nn.CrossEntropyLoss()
        # self.loss2 = nn.CrossEntropyLoss()
        self.max_grad_norm = 1 # magic number for now

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def _get_optimizer(self, loss_model):
        # Prepare optimizers
        param_optimizer = list(loss_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer_params = self.optimizer_params
        optimizer = self.optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        return optimizer

    def fit(self,
        CFG,
        train_loader,
        val_loader,
        loss_model,
        start_epoch=1,
        optimizer_class: Type[Optimizer] = AdamW,
        fstep = 0,
        loaded_optimizer_state = None,
        loaded_sched_state = None,
        ):
        self.text_model.to(self.device)
        self.audio_model.to(self.device)
        loss_model.to(self.device)

        self.val_loader = val_loader

        steps_per_epoch = len(train_loader)
        num_train_steps = int(steps_per_epoch * CFG.num_epochs)
        warmup_steps =  math.ceil(len(train_loader) *  CFG.num_epochs * 0.1)  
        self.weight_decay = CFG.weight_decay
        self.optimizer_class = optimizer_class
        self.optimizer_params = {'lr': CFG.lr}
        scheduler_method='WarmupLinear' 

        # print("[del] ", steps_per_epoch, num_train_steps, warmup_steps)
        steps_so_far = (start_epoch + 1) * fstep
        steps_per_epoch = len(train_loader)
        self.num_train_steps = num_train_steps

        # Initiate or load an optimizer
        if loaded_optimizer_state == None:
            optimizer = self._get_optimizer(loss_model)
        else:
            optimizer = self._get_optimizer(loss_model)
            # optimizer = loaded_optimizer
            optimizer.load_state_dict(loaded_optimizer_state)

        # Initiate or load a scheduler
        if loaded_sched_state == None:
            scheduler = self._get_scheduler(optimizer, scheduler=scheduler_method, warmup_steps=warmup_steps, t_total=num_train_steps)
        else:
            scheduler = self._get_scheduler(optimizer, scheduler=scheduler_method, warmup_steps=warmup_steps, t_total=num_train_steps)
            scheduler.load_state_dict(loaded_sched_state)

        # TODO: this is a test: does it help to learn the scale?
        # if loss_model.scale_type != 'learned':
        #     update_scale = False
        # else:
        #     update_scale = True
        # memory_test_delete = 1000

        for epoch in range(start_epoch, CFG.num_epochs):
            t1 = time()
            loss_model.zero_grad()
            loss_model.train()
            # Fixed length training:
            # iterator = iter(train_loader)
            # for step in range(self.total_len):
            #     batch = next(iterator)

            # Full training
            # TODO: if training from checkpoint, stop in time
            for step, batch in enumerate(iter(train_loader)):
                # print("[del] trainstep: ", steps_so_far, step)
                if step < fstep:
                    print("[debug] loading checkpoint, continue")
                    continue
                if steps_per_epoch == step:
                    print("[DEBUG] : remove -1 in line below!")

                if  step % self.eval_every == 0 or step == steps_per_epoch - 1: 
                    print("[eval] start evaluation")
                    mean_loss, metrics = self.evaluate(loss_model)
                    # mean_loss = 0 # TODO: check of mean loss terug kan komen uit eval
                    self.add_logging(epoch, steps_so_far, mean_loss, metrics, train=False)
                    
                    print("[eval] Epoch {} Step {}/{} \t loss {} \t acc {}".format(epoch, step, len(train_loader), mean_loss, metrics['mean_acc']))
                    if mean_loss < self.best_loss:
                        print("[eval] better model found")
                        self.best_loss = mean_loss 
                        if args.save_model:
                            self.save_model()
                        if args.save_checkpoint:
                            self.save_checkpoint(epoch, step, optimizer, scheduler)

                    # If csv files are properly set up, save plots.
                    if os.path.isfile(self.train_csv_filename):
                        self.output_all_plots()

                sent_features, audio_features, seq_len = batch
                loss_value, metrics = loss_model(sent_features, audio_features, seq_len)

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_model.parameters(), self.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                scheduler.step()
                steps_so_far += 1

                if step % self.print_every == 0:
                    print("[train] Epoch {} Step {}/{} \t loss {} \t acc {}".format(epoch, step, len(train_loader), loss_value.item(), metrics['mean_acc']))
                    self.add_logging(epoch, steps_so_far, loss_value.item(), metrics, train=True)
                
                # # [del] check on one batch
                # print("[del] [metrics] ", loss_value.item(), metrics['mean_acc'])

                # if step >= 0:
                #     print("[del] break")
                #     break   

                # # [del] check memory
                # if step % memory_test_delete == 0:
                #     process = psutil.Process(os.getpid())
                #     print("[del] [mem] training memory              : ", process.memory_info().rss)
                #     gc.collect()
                #     print("[del] [mem] training memory after collect: ", process.memory_info().rss)

            self.output_all_plots()
            t2 = time()
            print("[train] epoch duration {} seconds".format(int(t2-t1)))

        print("[fit] Done training")

    def evaluate(self, loss_model):
        # Set evaluation mode on
        # if args.test_evalchange:
        loss_model.eval()
        losses = 0
        # validation_len = len(self.test_loader)

        # Iterate over data and calculate accuracies
        self.to(self.device)

        full_validation = False
        with torch.no_grad():

            # fixed number of steps
            if not full_validation:
                iterator = iter(self.val_loader)

                total_len = 1     # 1000 bij test

                for step in range(total_len):
                    # print("[del] step: ", step)
                    batch = next(iterator)

                    sent_features, audio_features, seq_len  = batch
                    with torch.no_grad():
                        loss_value, metrics = loss_model(sent_features, audio_features, seq_len)
                        losses += loss_value.detach().cpu().item()
    
                        if step == 0:
                            #metrics_sum = metrics.copy()
                            met_sum = Counter(metrics.copy())
                        else:
                            #metrics_sum = {k: metrics_sum.get(k, 0) + metrics.get(k, 0) for k in set(metrics_sum)}
                            met_sum.update(Counter(metrics))

        mean_metrics = {k: value / total_len  for k, value in met_sum.items()}
        mean_loss = losses / total_len

        # print("[del] metrics: ", mean_metrics)
        del met_sum

        # if args.test_evalchange:
        loss_model.train()
        return mean_loss, mean_metrics

    def output_all_plots(self):
        to_plot(self.train_csv_filename, column='audio_acc', title="Train accuracy (audio)")
        to_plot(self.train_csv_filename, column='text_acc', title="Train accuracy (text)")
        to_plot(self.train_csv_filename, column='loss', title="Train loss")
        to_plot(self.eval_csv_filename, column='mean_acc', title="val accuracy (mean)")
  
    def init_logging(self):
        self.model_save_path = '{}/output'.format(self.log_name)
        os.makedirs(self.model_save_path, exist_ok=True)
        self.train_csv_filename = os.path.join(self.model_save_path, "train.csv")
        self.eval_csv_filename = os.path.join(self.model_save_path, "val.csv")

    def init_csv(self, metrics):
        for filename in [self.train_csv_filename, self.eval_csv_filename]:
            self.metric_headers = [metric for metric in metrics.keys()]
            self.train_csv_headers = ["epoch", "steps", "loss"] + self.metric_headers
            with open(filename, newline='', mode='w', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.train_csv_headers)

    def add_logging(self, epoch, steps, loss, metrics, train=True):
        if train:
            filename = self.train_csv_filename
        else:
            filename = self.eval_csv_filename

        output_file_exists = os.path.isfile(filename)
        if not output_file_exists:
            self.init_csv(metrics)
        total_step = (self.num_train_steps * epoch) + steps

        # Add metrics to CSV
        metric_vals = [metrics[header] for header in self.metric_headers]
        with open(filename, newline='', mode="a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, total_step, loss] + metric_vals)

    def save_model(self):
        # Debug: saves all models seperately
        # for idx, name in enumerate(self._modules):
        #     module = self._modules[name]
        #     print(module)
        #     output_dir = os.path.join(self.model_save_path, '{}_weights.pth'.format(name))
        #     torch.save(module.state_dict(), output_dir)

        # Save the model
        output_dir = os.path.join(self.model_save_path, '{}_weights.pth'.format('full_model'))
        torch.save(self.state_dict(), output_dir)

    def save_checkpoint(self, epoch, step, optimizer, scheduler):
        checkpoint = { 
            'epoch': epoch,
            'step': step,
            'full_model': self,
            'optimizer': optimizer,
            'lr_sched': scheduler.state_dict()
        }
        output_dir = os.path.join(self.model_save_path, 'checkpoint.pth')
        torch.save(checkpoint, output_dir)


def dek_cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))





########################### DELETE 
if __name__ == "__main__":
    print("[main]")
    model_path = "E:\msc_thesis\code\contrastive_mm2\logs\lisa_v2-simcse_loss_rnn_relu_768_2e-05_2022-05-17_06-58-44"
    model_path = '/Users/casper/Documents/UvAmaster/b23456_thesis/msc_thesis/code/contrastive_mm2/logs/windows_gru2-clip_loss_gru_gelu_768_5e-05_2022-05-26_21-51-25'
    model_path = '/Users/casper/Documents/UvAmaster/b23456_thesis/msc_thesis/code/contrastive_mm2/logs/windows_gru2-clip_loss_followup'
    # model_path = "E:\msc_thesis\code\contrastive_mm2\logs\windows_gru2-clip_loss_followup"

    # transcripts_path = '/Users/casper/Documents/UvAmaster/b23456_thesis/msc_thesis/code/data/sp/podcasts-no-audio-13GB/podcasts-transcripts'
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

    
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_model_name)


    # Create dataloader of test set
    data_loader = MMloader(CFG)
    # train_loader = data_loader.train_loader
    # val_loader = data_loader.val_loader
    test_loader = data_loader.test_loader # Not used!

    # TODO: move this into __init__ of dataloader
    data_loader.test_dataset.tokenizer = tokenizer
    data_loader.test_dataset.text_max_length = CFG.text_max_length

    # Load model and tokenizer
    full_model = mmModule(CFG)
    print("[del] device: ", CFG.device)
    full_model.load_state_dict(torch.load(model_weights,  map_location=CFG.device))              
    # full_model.load_state_dict(torch.load(model_weights))   
    full_model = full_model.to(CFG.device)     
    full_model.eval()



    do_vali = False
    create_embedding = True

    if do_vali:
        vali_accs = []
        iterator = iter(test_loader)
        max_steps = len(test_loader) 
        print("START TRAIN FOOR STEPS: ",  max_steps)

        for step in range(max_steps):
            print("step {}/{}".format(step, max_steps))
            batch = next(iterator)
            
            (tok_sentences, audio_features, seq_len, targets) = batch

            tok_sentences = tok_sentences.to(CFG.device)
            audio_features = audio_features.to(CFG.device)        

            with torch.no_grad():
                reps_sentences = full_model.text_model(tok_sentences)['sentence_embedding'].to(CFG.device)
                reps_audio = full_model.audio_model((audio_features, seq_len))
                
                audio_logits =  (reps_audio @ reps_sentences.t()) * CFG.scale
                text_logits = audio_logits.t()
                
                audio_logits = audio_logits / audio_logits.norm(dim=1, keepdim=True)
                text_logits = text_logits / text_logits.norm(dim=1, keepdim=True)
                
                probs = audio_logits.softmax(dim=-1).cpu().numpy()

                probs = audio_logits.softmax(dim=-1).cpu().numpy()
                ground_truth = torch.arange(128)
                acc = torch.eq(torch.tensor(probs.argmax(axis=0)), ground_truth).sum() / ground_truth.shape[0]
                print("accuracy", acc.item())
                vali_accs.append(acc.item())

            if step > 3:
                break
        print("TEST acc:{} ".format(np.mean(vali_accs)))


    if create_embedding:
        print("[Create embeds] start for TEST")
        iterator = iter(test_loader)
        max_steps = len(test_loader) 
        processed_text = []
        processed_audio = []

        topic_norm_reps_text = []
        topic_norm_reps_audio = []
        matrix_targets = []
        vali_accs = []

        # max_steps = 100 ## DEL

        sent_representations = np.zeros((max_steps * 128, 768)) # TODO: mutual embed dim
        audio_representations = np.zeros((max_steps * 128, 768))# TODO: mutual embed dim
		# image_representations = np.zeros((len(self.dataset.caption_ids), self.config.model.embed_dim))

        # This is for now to code faster, delete it
        my_tmp_file = Path("tmp_validation.hdf5")
        if my_tmp_file.is_file():
            f = h5py.File("tmp_validation.hdf5", "r")
            read_something = True
        else:
            f = h5py.File("tmp_validation.hdf5", "w")
            read_something= False
        # print("[del] this is keys: ", f.keys())
        for step in range(max_steps):
            print("{}/{}".format(step, max_steps))
            if step > 5:
                print('break for now')
                break
            batch = next(iterator)
            
            (tok_sentences, audio_features, seq_len, targets) = batch

            with torch.no_grad():

                all_targs_del = '-'.join(targets)
                # all_targs_del = targets[0]
                if not all_targs_del in f.keys():
                    #print("create it")
                    reps_sentences = full_model.text_model(tok_sentences)['sentence_embedding']
                    reps_audio = full_model.audio_model((audio_features, seq_len))

                    # dset = f.create_dataset(all_targs_del)
                    if not read_something:
                        grp = f.create_group(all_targs_del)
                        grp.create_dataset('reps_sentences', data=reps_sentences.cpu())
                        grp.create_dataset('reps_audio', data=reps_audio.cpu())

                else:
                    print("load it")
                    reps_sentences = torch.tensor(f[all_targs_del]['reps_sentences'])
                    reps_audio = torch.tensor(f[all_targs_del]['reps_audio'])

                # print("---Reps audio: ", type(reps_audio))
                # print("---Reps audio: ", (reps_audio.shape))
                
                # Normalized representations
                norm_reps_sentences = reps_sentences / reps_sentences.norm(dim=1, keepdim=True)
                norm_reps_audio = reps_audio / reps_audio.norm(dim=1, keepdim=True)

                # This will be returned
                idxs = np.arange(step * 128, step*128+128)
                sent_representations[idxs, :] = norm_reps_sentences.cpu().numpy().copy()
                audio_representations[idxs, :] = norm_reps_audio.cpu().numpy().copy()
                matrix_targets.extend(targets)

                # This has to be deleted
                # topic_norm_reps_text.append(reps_sentences / reps_sentences.norm(dim=1, keepdim=True))
                # topic_norm_reps_audio.append(reps_audio / reps_audio.norm(dim=1, keepdim=True))
                
                # Calculate accuracy of test set
                audio_logits =  (reps_audio @ reps_sentences.t()) * CFG.scale
                text_logits = audio_logits.t()
                
                probs = audio_logits.softmax(dim=-1).cpu().numpy()
                ground_truth = torch.arange(128)
                acc = torch.eq(torch.tensor(probs.argmax(axis=0)), ground_truth).sum() / ground_truth.shape[0]
                print("accuracy", acc.item())
                vali_accs.append(acc.item())

        print("This was acc: ", np.mean(vali_accs))

        # TODO: THis results in CUDA out oOOM
        # topic_norm_reps_audio = torch.cat(topic_norm_reps_audio, dim=0)
        # topic_norm_reps_text = torch.cat(topic_norm_reps_text, dim=0)
        # print(topic_norm_reps_audio.shape, topic_norm_reps_text.shape)
        # print(sent_representations.shape, audio_representations.shape)

        topic_norm_reps_text = sent_representations
        topic_norm_reps_audio = audio_representations
        print("[Create embeds] Done")
        

        ### Other func
        ### Validation DF
        ids_we_can_use = [m.split('_')[0] for m in matrix_targets]
        print("WE CAN USE {} ids".format(len(set(ids_we_can_use))))
        val_df = topics_df_targets[topics_df_targets['episode_uri'].isin(ids_we_can_use)]

        ### TODO: dit naar boven
        positive_episodes = val_df.loc[val_df['bin_score'] == 1].copy()
        positive_eplist = positive_episodes['episode_uri'].tolist()

        for i, row in val_df.iterrows():
            ifor_val = 0
            if row['episode_uri'] in positive_eplist:
                ifor_val = 1
            val_df.loc[i,'ep_score'] = ifor_val
            
        val_df.ep_score = val_df.ep_score.astype(int)
        print(len(val_df))

        print("Tokenizing queries")
        query_field =  'query' # TODO: ook description?
        queries = topics_df[query_field].tolist()
        tokenized_queries = tokenizer(
            queries, padding=True, truncation=True, max_length=32, return_tensors='pt', return_token_type_ids=True,
        ).to(CFG.device)

        print("Creating padding of yamnets of queries")
        query_yamnets = []
        query_lengths = []
        for idx, row in topics_df.iterrows():
            query_num = row.num

            query_embed_path  = os.path.join(conf.yamnet_query_embed_path, str(query_num) + ".h5")
            inbetween_yamnet_embeds = pd.read_hdf(query_embed_path)

            query_lengths.append(len(inbetween_yamnet_embeds))
            tmp = torch.Tensor(inbetween_yamnet_embeds.values)

            query_yamnets.append(tmp)
            #lengths.append(len(example[1]))

        padded_query_yamnets = pad_sequence(query_yamnets, batch_first=True).to(CFG.device)

        print("Creating query embeddings now:")
        query_text_repr = []
        query_audio_repr = []
        query_field = 'query'

        with torch.no_grad():
            # print("[del] get embeds: ")
            reps_sentences = full_model.text_model(tokenized_queries)['sentence_embedding']
            reps_audio = full_model.audio_model((padded_query_yamnets, query_lengths))

            # audio_logits =  (reps_audio @ reps_sentences.t()) * CFG.scale
            # text_logits = audio_logits.t()

            # audio_logits = audio_logits / audio_logits.norm(dim=1, keepdim=True)
            # text_logits = text_logits / text_logits.norm(dim=1, keepdim=True)
            #probs = audio_logits.softmax(dim=-1).cpu().numpy()
            # query_text_logits = text_logits
            # query_audio_logits = audio_logits
            
            query_norm_reps_audio = reps_audio / reps_audio.norm(dim=1, keepdim=True)
            query_norm_reps_text = reps_sentences / reps_sentences.norm(dim=1, keepdim=True)

            query_text_repr.append(query_norm_reps_text)
            query_audio_repr.append(query_norm_reps_audio)
    

        query_audio_repr = torch.cat(query_audio_repr, dim=0).cpu()
        query_text_repr = torch.cat(query_text_repr, dim=0).cpu()


        print("Shapes: ", query_audio_repr.shape, query_audio_repr.shape)
        print(": ", topic_norm_reps_text.shape)
        print("----------")
        print("TOPK")
        k = 50
        pred_episodes = {}
        # query_norm_reps_audio = query_norm_reps_audio.cpu()
        # query_norm_reps_text = query_norm_reps_text.cpu()

        similarity = 100 * query_audio_repr @ topic_norm_reps_text.T
        probs = similarity.softmax(dim=-1).cpu()

        top_probs, top_labels = probs.topk(k, dim=-1)

        for row_idx in range(len(topics_df)):
            query = topics_df[query_field][row_idx]
            pred_ind = top_labels[row_idx].tolist()

            pred_epis = [matrix_targets[val].split('_')[0] for val in pred_ind]
            query_num = val_df['num'][row_idx]
            print("checking for {} and num {} ".format(row_idx, query_num))
            
            tmp = val_df[(val_df.num == query_num) & (val_df.ep_score==1)]
            tmp = tmp['episode_uri'].tolist()
            num_episodes_relevant = len(set(tmp))
            print("[del] Relevant episodes for Query: ", num_episodes_relevant)
                    
            ep_scores = []
            for episode_uri in pred_epis:
                ep_score_row = val_df.loc[val_df['episode_uri'] == episode_uri]
                if len(ep_score_row) > 0 and ep_score_row.num.values[0] == query_num:
                    ep_scores.append(1)
                else:
                    ep_scores.append(0)
            
            pred_episodes[query] = {}
            # pred_episodes[query]['episodes'] = [matrix_targets[val].split('_')[0] for val in pred_ind]
            pred_episodes[query]['ep_score'] = ep_scores
            
            pred_episodes[query]['prec@3'] = precision_at_k(ep_scores, 3)
            pred_episodes[query]['prec@10'] = precision_at_k(ep_scores, 10)
            pred_episodes[query]['prec@30'] = precision_at_k(ep_scores, 30)
            

            targets = [[1] * num_episodes_relevant + [0] * (len(ep_scores) - num_episodes_relevant)]
            print("input ndcg: ", targets)
            print("and also: ", ep_scores)
            ndcg_ep_score = ndcg_score(targets, [ep_scores], k=30)
            pred_episodes[query]['ndcg'] = ndcg_ep_score
            print("done query {}, p@10 {}, ndcg: {}".format(num, pred_episodes[query]['prec@10'], ndcg_ep_score))

    mets = ['prec@3','prec@10','prec@30','ndcg']

    mean_mets = {}
    for m in mets:
        mean_mets[m] = []
    for query, qval in pred_episodes.items():
        query_metrics = pred_episodes[query]
        for m in (mets):
            mean_mets[m].append(query_metrics[m])

    for k, v in mean_mets.items():
        print("Metric: {}    Mean {}".format(k, np.mean(v)))

    with open("results.json", "w") as f:
        # json.dump(pred_episodes, f, indent=4)
        json.dump(mean_mets, f, indent=4)













        