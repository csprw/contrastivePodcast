"""
Contrastive multimodal learning
Author: Casper Wortmann
Usage: python main.py
"""
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
from omegaconf import OmegaConf

# Load static configuration variables. 
conf = OmegaConf.load("./config.yaml")
print("[cudacheck] Is cuda available? ", torch.cuda.is_available())

################################################################################
# move to Utils
def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def set_logconfig():
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.INFO)

def get_log_name(args, dc):
    log_name = "gru2-{}_{}_{}_{}_{}_{}".format(args.loss_type, args.audio_proj_head, 
            args.audio_activation, args.final_projection_dim, dc.lr,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_name = os.path.join(args.log_dir, log_name)
    return log_name

def setup_config(args, dc, device='cpu'):
    # Set the seed and create output directories
    set_seed(args)
    log_name = get_log_name(args, dc)
    os.makedirs(log_name, exist_ok=True)

    # Create a configuration file from CL arguments. 
    config = vars(args)
    config['log_name'] = log_name
    config['device'] = device

    # For all other keys, use the standard the dataclass arguments.
    for field in dc.__dataclass_fields__:
        value = getattr(dc, field)
        if field not in config.keys():
            config[field] = value

    fullcfg = from_dict(data_class=Cfg, data=config)
    print("[del] DATA: ")
    pprint(fullcfg)

    # Save config.
    config_dir = os.path.join(log_name, 'config.json')
    with open(config_dir, "w") as file:
        json.dump(config, file, indent=4, sort_keys=True)
    return fullcfg, log_name

################################################################################
##################### Dataloaders
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
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs) 
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
        
        # Tokenize text
        text_embeds = self.tokenizer(
            text_embeds, padding=True, truncation=True, max_length=self.text_max_length, return_tensors='pt'
        ).to(self.device)

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


class multimodal_loss(nn.Module):
    """
        This loss expects as input a batch consisting of ... etc
    """
    def __init__(self, full_model, CFG, normalize=False):
        """
        test
        """
        super(multimodal_loss, self).__init__()

        self.text_model = full_model.text_model
        self.audio_model = full_model.audio_model

        self.normalize = normalize
        self.loss_type = CFG.loss_type
        self.scale_type = CFG.scale_type
        self.scale = CFG.scale

        if self.scale_type == 'fixed':
            self.fixed_scale = self.scale
        elif self.scale_type == 'learned':
            # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            # self.init_parameters_logtiscale()
            self.logit_scale = nn.Parameter(torch.log(torch.ones([]) * 100))
            self.logit_scale.requires_grad = True

        self.similarity_fct = dek_cos_sim
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.device = torch.device(CFG.device)
        self.batch_size = full_model.batch_size

    def init_parameters_logtiscale(self):
        # nn.init.normal_(self.token_embedding.weight, std=0.02)
        # nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], audio_features: Tensor, seq_len):
        # Get sentence Representations (shape [batchsize, 768])
        reps_sentences = self.text_model(sentence_features)['sentence_embedding']
        # print("[clipmodel] text_embeddings: ", reps_sentences[0])
        
        # Get Audio representations
        reps_audio = self.audio_model((audio_features, seq_len))

        # ['simcse_loss', 'clip_loss', 'clip_loss_simple'] 
        if self.loss_type == 'clip_loss':
            # Loss function from CLIP paper
            if self.scale_type == 'fixed':
                audio_logits =  (reps_audio @ reps_sentences.t()) * self.fixed_scale
            elif self.scale_type == 'learned':
                cur_logit_scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
                audio_logits =  (reps_audio @ reps_sentences.t()) * cur_logit_scale.exp()

            text_logits = audio_logits.t()

            if self.normalize:
                #print("TODO: versie waarin normalise naar boven zit!!!")

                audio_logits = audio_logits / audio_logits.norm(dim=1, keepdim=True)
                text_logits = text_logits / text_logits.norm(dim=1, keepdim=True)

            # ground_truth = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  
            ground_truth = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
            # total_loss = (self.cross_entropy_loss(audio_logits,ground_truth) + self.cross_entropy_loss(text_logits,ground_truth))/2

            # TODO: werkt dit beter?
            total_loss = F.cross_entropy(audio_logits,ground_truth, weight=None) + F.cross_entropy(text_logits.transpose(-1, -2), ground_truth, weight=None)

            #audio_acc = torch.mean((audio_logits.detach().argmax(dim=-1) == ground_truth).float()).item()
            #text_acc = torch.mean((text_logits.detach().argmax(dim=-1) == ground_truth).float()).item()

            metrics = get_metrics(audio_logits.detach(), text_logits.detach(), ground_truth)
   
            return total_loss, metrics
    
        if self.loss_type == 'clip_loss_simple':
            # Loss function from CLIP paper
            if self.scale_type == 'fixed':
                # temp = 40.0
                logits =  (reps_sentences @ reps_audio.t()) / self.fixed_scale
                audio_logits = reps_audio @ reps_audio.T
                text_logits = reps_sentences @ reps_sentences.T
                ground_truth = F.softmax((audio_logits + text_logits) / 2 * self.fixed_scale, dim=-1)

            elif self.scale_type == 'learned':
                cur_logit_scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
                logits =  (reps_sentences @ reps_audio.t()) / cur_logit_scale.exp()
                audio_logits = reps_audio @ reps_audio.T
                text_logits = reps_sentences @ reps_sentences.T
                ground_truth = F.softmax((audio_logits + text_logits) / 2 * cur_logit_scale.exp(), dim=-1)
                
            if self.normalize:
                audio_logits = audio_logits / audio_logits.norm(dim=1, keepdim=True)
                text_logits = text_logits / text_logits.norm(dim=1, keepdim=True)

            texts_loss = cross_entropy(logits, ground_truth, reduction='none')
            images_loss = cross_entropy(logits.T, ground_truth.T, reduction='none')
            total_loss =  ((images_loss + texts_loss) / 2.0).mean() 
 
            #audio_acc = torch.mean((audio_logits.argmax(dim=-1) == ground_truth).float()).item()
            #text_acc = torch.mean((text_logits.argmax(dim=-1) == ground_truth).float()).item()
            metrics = get_metrics(audio_logits.detach(), text_logits.detach(), ground_truth)
            return total_loss, metrics

        elif self.loss_type == 'simcse_loss':
            # SIMCSE-based NCE loss
            if self.scale_type == 'fixed':
                # audio_logits =  (reps_audio @ reps_sentences.t()) * self.fixed_scale
                scores = self.similarity_fct(reps_sentences, reps_audio) * self.fixed_scale
            elif self.scale_type == 'learned':
                cur_logit_scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
                # audio_logits =  (reps_audio @ reps_sentences.t()) * self.logit_scale.exp()
                scores = self.similarity_fct(reps_sentences, reps_audio) * cur_logit_scale.exp()

            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
            loss = self.cross_entropy_loss(scores, labels)

            #acc = torch.mean((scores.argmax(dim=-1) == labels).float()).item()

            metrics = get_metrics(scores.detach(), scores.t().detach(), labels)
            return loss, metrics

    def get_config_dict(self):
        return {
            'scale_type': self.scale_type, 
            'similarity_fct': self.similarity_fct.__name__
        }

def get_metrics(audio_logits, text_logits, ground_truth):
    # tODO: naar self.multimodal_loss
    metrics ={}
    logits = {"audio": audio_logits, "text": text_logits}

    accs = []
    for name, logit in logits.items():
        acc = torch.mean((logit.argmax(dim=-1) == ground_truth).float()).item()
        accs.append(acc)
        metrics[f"{name}_acc"] = acc
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        if len(preds) > 0:
            metrics[f"{name}_mean_rank"] = preds.mean() + 1
            #metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1

            for k in [1, 5, 10]:
                metrics[f"{name}_R@{k}"] = np.mean(preds < k)
        else:
            metrics[f"{name}_mean_rank"] = 0.0
            for k in [1, 5, 10]:
                metrics[f"{name}_R@{k}"] = 0.0
    metrics['mean_acc'] = np.mean(accs)
    return metrics

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def to_plot(filename, column='accuracy', title="Val accuracy"):
    csv_file = pd.read_csv(filename)

    ax = sns.lineplot(x=csv_file.steps, y=csv_file[column])
    ax.set(title=title)
    
    output_dir = os.path.split(filename)[0]
    output_title = os.path.split(filename)[-1].split('.')[0]
    output_name = os.path.join(output_dir, output_title + "_" + column + '.jpg')
    plt.savefig(output_name)
    plt.close()

            
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


######### This stays in MAIN
def main(args):
    set_logconfig()
    t_start = time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Change the CFG file according to argparse.
    # FullCfg.num_epochs = args.num_epochs
    # FullCfg.final_projection_dim = args.final_projection_dim
    # FullCfg.audio_proj_head = args.audio_proj_head
    # FullCfg.text_proj_head = args.text_proj_head
    # FullCfg.audio_activation = args.audio_activation
    # FullCfg.text_activation = args.text_activation
    # FullCfg.device = device
    # FullCfg.batch_size = args.batch_size
    # FullCfg.loss_type = args.loss_type
    # FullCfg.text_pooling = args.text_pooling
    # FullCfg.scale_type = args.scale_type
    # FullCfg.scale = args.scale


    FullCfg, log_name = setup_config(args, Cfg, device)

    # Setup dataloaders.
    data_loader = MMloader(FullCfg)
    train_loader = data_loader.train_loader
    val_loader = data_loader.val_loader
    test_loader = data_loader.test_loader # Not used!

    # tokenizer = AutoTokenizer.from_pretrained(FullCfg.text_model_name, cache_dir=None)
    tokenizer = AutoTokenizer.from_pretrained(FullCfg.text_model_name)

    # TODO: move this into __init__ of dataloader
    data_loader.train_dataset.tokenizer = tokenizer
    data_loader.val_dataset.tokenizer = tokenizer
    data_loader.test_dataset.tokenizer = tokenizer
    data_loader.train_dataset.text_max_length = FullCfg.text_max_length
    data_loader.val_dataset.text_max_length = FullCfg.text_max_length
    data_loader.test_dataset.text_max_length = FullCfg.text_max_length

    # Evaluate every 10% of the data, print result every 2%.
    FullCfg.eval_every = int(math.ceil(len(train_loader) * 0.1)) 
    FullCfg.print_every = int(math.ceil(len(train_loader) * 0.02))
    print("[main] print_every {} eval_every {} ".format(
        FullCfg.print_every, FullCfg.eval_every)
    )

    # Setup the full model
    full_model = mmModule(FullCfg)

    # Setup the loss function
    normalize=args.normalize             # TODO: dit kan weg in zijn geheel
    loss_func = multimodal_loss(full_model, FullCfg, normalize)


    if args.load_checkpoint:
        print("[Main] train model {} from checkpoint".format(
            args.load_checkpoint_path)
        )
        ####
        # TODO: fix this! to read file if trained on windows
        import pathlib
        print(sys.platform)
        plt = sys.platform
        if plt != 'win32': 
            pathlib.WindowsPath = pathlib.PosixPath
        #####

        checkpoint = torch.load(args.load_checkpoint_path, map_location=torch.device(device))
        # print("checkpoint: ", checkpoint)
        epoch = checkpoint['epoch']
        fstep = checkpoint['step']
        # full_model = checkpoint['full_model'].state_dict()
        full_model.load_state_dict(checkpoint['full_model'].state_dict())

        # full_model.load_state_dict(checkpoint['full_model_state_dict'])
        loaded_optimizer_state = checkpoint['optimizer'].state_dict()
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loaded_sched_state = checkpoint['lr_sched']

        full_model.device = FullCfg.device
        print("------ CHECK LOADED")

    else:
        if args.load_model: 
            print("[Main] load model {}, but initiate new optimizer".format( 
                args.load_model_path)
            )
            full_model.load_state_dict(torch.load(args.load_model_path))
        epoch = 0
        fstep = 0
        loaded_optimizer_state = None
        loaded_sched_state = None
        
    full_model.fit(
        CFG = FullCfg,
        train_loader=train_loader,
        val_loader = val_loader,
        loss_model=loss_func,
        start_epoch=epoch,
        optimizer_class=AdamW,
        fstep = fstep,
        loaded_optimizer_state = loaded_optimizer_state,
        loaded_sched_state = loaded_sched_state
    )

    t_end = time()
    print("[main] ------------------------------------------------------------")
    dur = int(t_end - t_start)
    print("[main] Done, total duration {} seconds ".format(dur))
    # csv_file = pd.read_csv(full_model.eval_csv_filename)
    # print("[main] Maximum text acc step={} acc={}".format(csv_file['text_acc'].idxmax(), csv_file['text_acc'].max()))
    # print("[main] Maximum audio acc step={} acc={}".format(csv_file['audio_acc'].idxmax(), csv_file['audio_acc'].max()))
    # best_idx = csv_file['mean_acc'].idxmax()
    # print("[main] Maximum mean acc step={} acc={}".format(best_idx, csv_file['mean_acc'].max()))
    # print(", ".join(["{} - {}".format(k, v) for k, v in csv_file.iloc[best_idx].to_dict().items()]))

    # print("[del] now save to disk")
    best_results(full_model.eval_csv_filename, dur, FullCfg.log_name)


def best_results(eval_csv_filename, dur, out_dir):
    # print("[del] this is eval_csv_filename: ", eval_csv_filename)

    csv_file = pd.read_csv(eval_csv_filename)
    best_idx = csv_file['mean_acc'].idxmax()
    best2 = {'duration': dur}
    for k, v in csv_file.iloc[best_idx].to_dict().items():
        best2[k] = v

    outfile = os.path.join(out_dir, 'best_results2.json')
    with open(outfile, "w") as file:
        json.dump(best2, file, indent=4, sort_keys=True)
    print("---- Best epoch results ----")
    pprint(best2)



if __name__ == "__main__":
    # Parse flags in command line arguments
    parser = ArgumentParser()

    parser.add_argument('--train_dataset', default='sp_sample', const='sp_sample',
                    nargs='?', choices=['sp_sample', 'sp'],
                    help='Name of training dataset (default: %(default)s)')
    parser.add_argument('--val_dataset', default='sp_sample', const='sp_sample',
                    nargs='?', choices=['sp_sample',  'sp'],
                    help='Name of validation dataset (default: %(default)s)')
    parser.add_argument('--test_dataset', default='sp_sample', const='sp_sample',
                    nargs='?', choices=['sp_sample',  'sp'],
                    help='Name of test dataset (default: %(default)s)')

    parser.add_argument('--seed', type=int, default=100,
                        help='Seet to use')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs to train')
    

    parser.add_argument('--log_dir', type=str, default="./logs",
                        help='Folder where so save logs.')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='batch_size')

    parser.add_argument('--final_projection_dim', type=int, default=768, 
                    nargs='?', choices=[256, 768],
                    help='Final output dimensions of the embeddings')

    # parser.add_argument('--clip_loss', action='store_true')
    parser.add_argument('--loss_type', default='simcse_loss', const='simcse_loss',
                    nargs='?', choices=['simcse_loss', 'clip_loss', 'clip_loss_simple'],
                    help='Name of scale_type (default: %(default)s)')

    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--audio_proj_head', default='gru', const='gru',
                    nargs='?', choices=['simple_projection_head', 'rnn', 'gru'],
                    help='Activation to use in simple proj head (default: %(default)s)')
    parser.add_argument('--text_proj_head', default='None', const='None',
                    nargs='?', choices=['simple_projection_head', 'None'],
                    help='Activation to use in simple proj head (default: %(default)s)')
    parser.add_argument('--text_pooling', default='mean', const='mean',
                    nargs='?', choices=['cls', 'mean'],
                    help='Pooling method to use for text model (default: %(default)s)')

    parser.add_argument('--audio_activation', default='relu', const='relu',
                    nargs='?', choices=['relu', 'gelu'],
                    help='Activation to use in simple proj head (default: %(default)s)')
    parser.add_argument('--text_activation', default='gelu', const='gelu',
                    nargs='?', choices=['relu', 'gelu'],
                    help='Activation to use in simple proj head (default: %(default)s)')
    parser.add_argument('--scale_type', default='fixed', const='fixed',
                    nargs='?', choices=['fixed', 'learned'],
                    help='Name of scale_type (default: %(default)s)')
    parser.add_argument('--scale', type=float, default=20,
                        help='Fixed scale to use')

    # GPU device number as in "cuda:0". Defaul is 0.
    parser.add_argument("-cuda", "--cuda_number", type=str, default='0')
    # parser.add_argument('--eval_every', type=int, default=100)

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # parser.add_argument('--fp16', action='store_true', default=False,
    #                     help='enables fp16 training')

    # parser.add_argument('--use_lr', dest='use_lr', action='store_true',
    #                     help="Change the learning rate")

    parser.add_argument('--save_model', dest='save_model', action='store_true',
                        help="Save the model weights.")
    parser.add_argument('--load_model_path', type=str, default="./logs/load_test2/output/full_model_weights.pth",
                        help='Folder where model weights are saved.')
    parser.add_argument('--load_model', dest='load_model', action='store_true',
                        help="Load the model in model_path to continue a downstream task")

    parser.add_argument('--save_checkpoint', dest='save_checkpoint', action='store_true',
                        help="Save the model, optimizer and scheduler weights.")
    parser.add_argument('--load_checkpoint_path', type=str, default="./logs/load_test2/output/checkpoint.pth",
                        help='Folder where so save logs.')
    parser.add_argument('--load_checkpoint', dest='load_checkpoint', action='store_true',
                        help="Load a model, optimizer and scheduler to continue training.")

    # parser.add_argument('--use_old_opt', dest='use_old_opt', action='store_true',
    #                     help="test: use old or new opt.")
    # parser.add_argument('--test_evalchange', dest='test_evalchange', action='store_true',
    #                     help="test: use old or new opt.")
    parser.add_argument('--pad_pack', dest='pad_pack', action='store_true',
                        help="test: use old or new opt.")
    args, unparsed = parser.parse_known_args()

    main(args)