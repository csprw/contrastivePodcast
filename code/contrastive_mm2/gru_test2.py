"""
Contrastive multimodal learning
Author: Casper Wortmann
Usage: python main.py
"""
from doctest import OutputChecker
from email.mime import audio
import logging
from argparse import ArgumentParser
import itertools
import sched 

import time
from time import time
from datetime import datetime
import h5py
import json, csv
import os
import psutil, gc
from pathlib import Path
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter, OrderedDict
from pprint import pprint

from omegaconf import OmegaConf
import gzip
from typing import List, Dict, Optional, Union, Tuple, Iterable, Type, Callable
from dataclasses import dataclass, fields, asdict
from tqdm.autonotebook import trange

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

def get_log_name(args, dataclasses):
    log_name = "short-{}_{}_{}_{}_{}_{}".format(args.loss_type, args.audio_proj_head, 
            args.audio_activation, args.final_projection_dim, dataclasses[0].head_lr,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_name = os.path.join(args.log_dir, log_name)
    return log_name

def setup_config(args, dataclasses):
    # Set the seed and create output directories
    set_seed(args)
    log_name = get_log_name(args, dataclasses)
    os.makedirs(log_name, exist_ok=True)

    # Create a configuration file
    config = vars(args)
    for dc in dataclasses:
        name = dc.__name__
        config[name] = {}
        for field in dc.__dataclass_fields__:
            value = getattr(dc, field)
            config[name][field] = value

    print("Config used: ")
    pprint(config)
    config_dir = os.path.join(log_name, 'config.json')
    with open(config_dir, "w") as file:
        json.dump(config, file, indent=4, sort_keys=True)
    return log_name
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
    def __init__(self, args, CFG, kwargs={}):

        train_dataset_name = args.train_dataset
        dev_dataset_name = args.dev_dataset
        test_dataset_name = args.test_dataset

        batch_size = args.batch_size
        self.batch_size = batch_size
        self.device = CFG.device

        if args.audio_proj_head in ['gru', 'rnn']:
            self.load_full = True
        else:
            self.load_full = False

        print("[MMloader] train dataset ", train_dataset_name)

        # Get the datasets
        if train_dataset_name == "wiki":
            train_dataset = self.get_wiki_dataset()
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        elif train_dataset_name == "sp_sample":
            train_dataset = self.get_sp_dataset(directory=conf.sp_sample_path, traintest="train", load_full=self.load_full, device=self.device)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)  ### TODO: SHUFFLE=TRUE [DEL]
        elif train_dataset_name == "sp":
            train_dataset = self.get_sp_dataset(directory=conf.sp_path,  traintest="train", load_full=self.load_full, device=self.device)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
        else:
            raise Exception('Unknown dataset')
        
        self.train_dataset = train_dataset
        self.train_loader.collate_fn = self.train_dataset.collate_fn
        print("[MMloader] train dataset loaded, length: ", len(self.train_dataset))

        if test_dataset_name == "sts":
            sts_dataset = self.get_sts_dataset()
            test_samples = sts_dataset.test_samples
        elif test_dataset_name == "sp_sample":
            test_dataset = self.get_sp_dataset(directory=conf.sp_sample_path,  traintest="test", load_full=self.load_full, device=self.device)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs) 
        elif test_dataset_name == "sp":
            test_dataset = self.get_sp_dataset(directory=conf.sp_path,  traintest="test",  load_full=self.load_full, device=self.device)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
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

        if self.load_full:
            self.collate_fn = self.full_batching_collate
        else:
            self.collate_fn = self.mean_batching_collate

    def __len__(self):
        """ Denotes the total number of utterances """
        return len(self.idx2file.keys())

    def __getitem__(self, index):
        """ Return one item from the df """
        if self.load_full:
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
        
        return padded_audio_embeds.float(), lengths, text_embeds

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
        text_embeds = self.tokenizer(
            text_embeds, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt'
        ).to(self.device)
        
        return audio_embeds, lengths, text_embeds

################################################################################
# Textmodules
class TextEncoder(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        # if CFG.pretrained:
        #     self.model = DistilBertModel.from_pretrained(CFG.text_encoder_model)
        # else:
        #     self.model = DistilBertModel(config=DistilBertConfig())
            
        # for p in self.model.parameters():
        #     p.requires_grad = CFG.trainable

        model_name_or_path = CFG.text_model_name
        self.config_keys = ['max_seq_length', 'do_lower_case']
        

        model_args: Dict = {}
        self.do_lower_case = None
        my_dropout = 0.1
        cache_dir = None
        config_kwargs = {
            "attention_dropout": my_dropout,
            "dropout": my_dropout,
        }

        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, **config_kwargs)
        self.hidden_dim = config.dim
        print("[Transformer_multimodal] Configurations used: \n", config)
        
        # This is self._load_model in mm2
        self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

        self.max_seq_length = CFG.text_max_length


        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0
        self.pooling = CFG.text_pooling

    def forward(self, text_embeds):
        # print("nieuwe poging, in forward: ")
        # input_ids=text_embeds["input_ids"]
        # attention_mask=text_embeds["attention_mask"]
        # print(input_ids,attention_mask )
        # hiero
        ######### LEAVE FOR DEBUG
        # Forward passes with different pooling strategies. 
        # if self.pooling == 'original':
        #     # This is what it was OG
        #     output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        #     last_hidden_state = output.last_hidden_state
        #     return last_hidden_state[:, self.target_token_idx, :]


        # elif self.pooling == 'cls':
        #     output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        #     output_tokens = output[0]
        #     # features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']}) # dit heb ik dus al
        #     cls_token = output_tokens[:, 0]  # Take first token by default
        #     return cls_token

        # elif self.pooling == 'mean':
        #     #print("[mean] pooling")
        #     # This is same as mm2
        #     print("[del] before output! input_ids :", input_ids)
        #     print("[del] before output! input_ids :", attention_mask)
        #     output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        #     print("[del] thorugh model ", type(output))
        #     exit(1)
        #     output_tokens = output[0]

        #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(output_tokens.size()).float()
        #     sum_embeddings = torch.sum(output_tokens * input_mask_expanded, 1)

        #     sum_mask = input_mask_expanded.sum(1)
        #     sum_mask = torch.clamp(sum_mask, min=1e-9)
        #     pooled_output = sum_embeddings / sum_mask

        #     return pooled_output

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
        self.layer_dim = CFG.layer_dim
        self.device = CFG.device
        self.audio_model = CFG.audio_proj_head

        # RNN layers
        if self.audio_model == 'rnn':
            self.seq_model = nn.RNN(CFG.audio_encoder_input, 
                    CFG.audio_hidden_dim, CFG.layer_dim, 
                    batch_first=True, dropout=CFG.audio_dropout)
        elif self.audio_model == 'gru':
            self.seq_model = nn.GRU(
                    input_size=CFG.audio_encoder_input, 
                    hidden_size=CFG.audio_hidden_dim, num_layers=CFG.layer_dim, 
                    batch_first=True, dropout=CFG.audio_dropout)

        # Fully connected layer
        self.fc = nn.Linear(CFG.audio_hidden_dim, CFG.mutual_embedding_dim)
        self.softmax = nn.Softmax(dim=1)

        pad_pack = False
        self.pad_pack = pad_pack
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
                print("PACK PAD ON!!")
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
                nn.Dropout(),
                nn.LayerNorm(self.output_dim),
            )

    def __repr__(self):
        return "Simple_projection_head({}) ".format(self.get_config_dict())
    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def forward(self, features, lengths):
        """Returns token_embeddings, cls_token"""
        features = self.simple_model(features)

        return features

class ProjectionHead(nn.Module):
    def __init__(self, CFG):

        super().__init__()
        embedding_dim=CFG.mutual_embedding_dim
        projection_dim=CFG.final_projection_dim
        dropout=CFG.text_dropout

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CLIPModel(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        temperature=CFG.temperature
        self.batch_size = CFG.batch_size
        self.device = CFG.device

        self.loss_type = CFG.loss_type

        # TODO: TEXT PROJECTION EENS UITZETTEN
        if CFG.text_proj_head == 'simple_projection_head':
            self.text_encoder = TextEncoder(CFG)
            self.text_projection = ProjectionHead(CFG)

            text_modules = [self.text_encoder, self.text_projection]
        elif CFG.text_proj_head.lower() == 'none':
            text_encoder = TextEncoder(CFG)
            text_projection = None

            pooling_model = Pooling(text_encoder.get_word_embedding_dimension())
            text_modules = [text_encoder, pooling_model]

        # old
        # if CFG.audio_proj_head in ['rnn', 'gru']:
        #     self.audio_encoder = SequentialAudioModel(CFG)
        #     self.audio_projection = ProjectionHead(CFG)
        #     audio_modules = [self.audio_encoder, self.audio_projection]
        # else: 
        #     self.audio_encoder = simple_ProjectionHead(CFG)
        #     self.audio_projection = None
        #     audio_modules = [self.audio_encoder]

        if CFG.audio_proj_head in ['rnn', 'gru']:
            audio_encoder = SequentialAudioModel(CFG)
            audio_modules = [audio_encoder]
        
        # Create the full model
        if text_modules is not None and not isinstance(text_modules, OrderedDict):
            text_modules = OrderedDict([(str(idx), module) for idx, module in enumerate(text_modules)])
        if audio_modules is not None and not isinstance(audio_modules, OrderedDict):
            audio_modules = OrderedDict([(str(idx), module) for idx, module in enumerate(audio_modules)])

        self.temperature = temperature
        
        self.text_model = nn.Sequential(text_modules)
        self.audio_model = nn.Sequential(audio_modules)

        if self.loss_type == 'clip_loss_simple':
            self.loss1 = nn.CrossEntropyLoss()
            self.loss2 = nn.CrossEntropyLoss()

        if self.loss_type == 'simcse_loss':
            self.fixed_scale = 20.0
            self.similarity_func = dek_cos_sim
            self.cross_entropy_loss = nn.CrossEntropyLoss()
        
    def forward(self, batch):
        audio_embeddings, lengths, text_embeds  = batch

        # Get  sentence Representations (shape [batchsize, 768])
        text_embeddings = self.text_model(text_embeds)['sentence_embedding']
        # print("[clipmodel] text_embeddings: ", text_embeddings[0])
        audio_embeddings = self.audio_model((audio_embeddings, lengths))
        # print("[clipmodel] audio_embeddings: ", audio_embeddings[0][0])
        # print("------------------")

        # Calculating the Loss
        if self.loss_type == 'clip_loss':
            logits = (text_embeddings @ audio_embeddings.T) / self.temperature
            audio_similarity = audio_embeddings @ audio_embeddings.T
            texts_similarity = text_embeddings @ text_embeddings.T
            targets = F.softmax(
                (audio_similarity + texts_similarity) / 2 * self.temperature, dim=-1
            )
            texts_loss = cross_entropy(logits, targets, reduction='none')
            audio_loss = cross_entropy(logits.T, targets.T, reduction='none')
            loss =  (audio_loss + texts_loss) / 2.0 # shape: (batch_size)

            labels = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
            metrics = get_metrics(logits.detach(), logits.t().detach(), labels)
        elif self.loss_type == 'clip_loss_simple':
            text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)
            audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=1)

            labels = torch.arange(self.batch_size, dtype=torch.long, device=self.device)

            audio_logits =  (audio_embeddings @ text_embeddings.t())
            text_logits = audio_logits.t()

            loss = (self.loss1(audio_logits, labels) + self.loss2(text_logits, labels))/2
            metrics = get_metrics(audio_logits.detach(), text_logits.detach(), labels)

        elif self.loss_type == 'simcse_loss':
            # print(text_embeddings.shape, audio_embeddings.shape)
            scores = self.similarity_func(text_embeddings, audio_embeddings) * self.fixed_scale
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]

            loss = self.cross_entropy_loss(scores, labels)
            metrics = get_metrics(scores.detach(), scores.t().detach(), labels)

        return loss, metrics

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


def get_metrics(audio_logits, text_logits, ground_truth):
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

def to_plot(filename, column='accuracy2', title="Test accuracy"):
    csv_file = pd.read_csv(filename)

    ax = sns.lineplot(x=csv_file.steps, y=csv_file[column])
    ax.set(title=title)
    
    output_dir = os.path.split(filename)[0]
    output_title = os.path.split(filename)[-1].split('.')[0]
    output_name = os.path.join(output_dir, output_title + "_" + column + '.jpg')
    plt.savefig(output_name)
    plt.close()

class Optimization:
    def __init__(self, fullCFG, model):
        self.model = model
        # self.lr_scheduler = lr_scheduler
        # self.optimizer = optimizer

        self.eval_every = fullCFG.eval_every
        self.print_every = fullCFG.print_every
        self.best_loss = float('inf')
        self.log_name = fullCFG.log_name
        self.init_logging()
        self.batch_size = fullCFG.batch_size
        self.device = fullCFG.device

        self.total_len = 1000
        self.total_len = 100

        self.loss1 = nn.CrossEntropyLoss()
        self.loss2 = nn.CrossEntropyLoss()

        self.max_grad_norm = 1 # magic number for now

    def train_epoch(self, epoch, train_loader, val_loader):
        self.model.train()
        self.model.zero_grad()
        steps = len(train_loader)
        total_steps = epoch * steps
        t1 = time()

        memory_test_delete = 1000

        # Fixed length training:
        # iterator = iter(train_loader)
        # for step in range(self.total_len):
        #     batch = next(iterator)

        # Full training
        for step, batch in enumerate(iter(train_loader)):
            # print("[del]: continue eval")
            if  step % self.eval_every == 0 and step < 1: # del last statement
                # Todo: uncommen
                mean_loss, metrics = self.evaluate(val_loader)
                self.add_logging(epoch, total_steps, mean_loss, metrics, train=False)
                
                print("[eval] Epoch {} Step {}/{} \t loss {} \t acc {}".format(epoch, total_steps, steps, mean_loss, metrics['mean_acc']))
                # if mean_loss < self.best_loss:
                #     print("[eval] better model found")
                #     self.best_loss = mean_loss 
                #     if args.save_model:
                #         self.save_model()
                #     elif args.save_checkpoint:
                #         self.save_checkpoint(epoch, step)
                # if step > self.print_every:
                #     self.output_all_plots()

            #padded_audio_embeds, lengths, text_embeds  = batch
            loss, metrics = self.model(batch)

            #self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

            if step % self.print_every == 0:
                print("[train] Epoch {} Step {}/{} \t loss {} \t acc {}".format(epoch, total_steps, steps, loss.item(), metrics['mean_acc']))
                self.add_logging(epoch, total_steps, loss.item(), metrics, train=True)
            
            total_steps += 1
            print("[metrics] ", loss.item(), metrics['mean_acc'])
            if step >= 0:
                print("[del] break")
                break   
            


            # # [del]
            # if step % memory_test_delete == 0:
            #     process = psutil.Process(os.getpid())
            #     print("[del] [mem] training memory              : ", process.memory_info().rss)
            #     gc.collect()
            #     print("[del] [mem] training memory after collect: ", process.memory_info().rss)


                
        self.output_all_plots()
        t2 = time()
        print("[train] epoch duration {} seconds".format(int(t2-t1)))
        return metrics

    def evaluate(self, val_loader):
        self.model.eval()
        losses = []
        validation_len = len(val_loader)

        full_validation = False
        with torch.no_grad():

            # fixed number of steps
            if not full_validation:
                iterator = iter(val_loader)
                for step in range(self.total_len):
                    batch = next(iterator)
                    # padded_audio_embeds, lengths, text_embeds  = batch
                    loss, metrics = self.model(batch)
                    losses.append(loss.item())
                    if step == 0:
                        met_sum = Counter(metrics.copy())
                    else:
                        #metrics_sum = {k: metrics_sum.get(k, 0) + metrics.get(k, 0) for k in set(metrics_sum)}
                        met_sum.update(Counter(metrics))
                    
                    # TODO: uncomment
                    break
                mean_metrics = {k: value / self.total_len  for k, value in met_sum.items()}
                del met_sum

                mean_loss = np.mean(losses)
                self.model.train()
                return mean_loss, mean_metrics

            
            else: 
                # full learning
                for step, batch in enumerate(iter(val_loader)):
                    # padded_audio_embeds, lengths, text_embeds  = batch
                    loss, metrics = self.model(batch)
                    losses.append(loss.item())
                    if step == 0:
                        met_sum = Counter(metrics.copy())
                    else:
                        #metrics_sum = {k: metrics_sum.get(k, 0) + metrics.get(k, 0) for k in set(metrics_sum)}
                        met_sum.update(Counter(metrics))
                        
                mean_metrics = {k: value / validation_len  for k, value in met_sum.items()}
                mean_loss = np.mean(losses)
                self.model.train()
                return mean_loss, mean_metrics

    def train(self, train_loader, val_loader, startepoch=0):
        # self.model.audio_encoder.to(self.device)
        # self.model.text_encoder.to(self.device)
        # if self.model.audio_projection:
        #     self.model.audio_projection.to(self.device)
        # if self.model.text_projection:
        #     self.model.text_projection.to(self.device)

        self.model.text_model.to(self.device)
        self.model.audio_model.to(self.device)

        steps_per_epoch = len(train_loader)
        num_train_steps = int(steps_per_epoch * FullCfg.num_epochs)

        warmup_steps =  math.ceil(len(train_loader) *  FullCfg.num_epochs * 0.1)  # 10% of train data for warm-up

        optimizer = self._get_optimizer(self.model)

        scheduler_method='WarmupLinear'
        scheduler = self._get_scheduler(optimizer, scheduler=scheduler_method, warmup_steps=warmup_steps, t_total=num_train_steps)
        
        self.optimizer = optimizer
        self.lr_scheduler = scheduler


        for epoch in range(startepoch, FullCfg.num_epochs):
            self.train_epoch(epoch, train_loader, val_loader)

        #mean_loss, eval_metrics = self.evaluate(val_loader)
        #self.add_logging(epoch, 0, mean_loss, eval_metrics, train=False)
        self.output_all_plots()

    def output_all_plots(self):
        to_plot(self.train_csv_filename, column='audio_acc', title="Train accuracy (audio)")
        to_plot(self.train_csv_filename, column='text_acc', title="Train accuracy (text)")
        to_plot(self.train_csv_filename, column='loss', title="Train loss")
        to_plot(self.eval_csv_filename, column='mean_acc', title="Test accuracy (mean)")
  
    def init_logging(self):
        self.model_save_path = '{}/output'.format(self.log_name)
        os.makedirs(self.model_save_path, exist_ok=True)
        self.train_csv_filename = os.path.join(self.model_save_path, "train.csv")
        self.eval_csv_filename = os.path.join(self.model_save_path, "test.csv")

    def add_logging(self, epoch, steps, loss, metrics, train=True):
        if train:
            filename = self.train_csv_filename
        else:
            filename = self.eval_csv_filename
        total_step = (self.total_len * epoch) + steps
        output_file_exists = os.path.isfile(filename)

        if not output_file_exists:
            self.metric_headers = [metric for metric in metrics.keys()]
            self.train_csv_headers = ["epoch", "steps", "loss"] + self.metric_headers
            with open(filename, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.train_csv_headers)

        # Add metrics to CSV
        metric_vals = [metrics[header] for header in self.metric_headers]
        with open(filename, newline='', mode="a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, total_step, loss] + metric_vals)

    def save_model(self):
        output_dir = os.path.join(self.model_save_path, 'full_model_weights.pth')
        torch.save(self.model.state_dict(), output_dir)

    def save_checkpoint(self, epoch, step):
        checkpoint = { 
            'epoch': epoch,
            'step': step,
            'full_model': self.model,
            'optimizer': self.optimizer,
            'lr_sched': self.lr_scheduler
        }
        output_dir = os.path.join(self.model_save_path, 'checkpoint.pth')
        torch.save(checkpoint, output_dir)

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
        self.weight_decay: float = 0.01 # todo: naar self of CFG
        self.optimizer_params : Dict[str, object]= {'lr': 5e-5}
        self.optimizer_class: Type[Optimizer] = AdamW
        
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

        
#     # @staticmethod
# def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
#     """
#     Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
#     """
#     scheduler = scheduler.lower()
#     if scheduler == 'constantlr':
#         return get_constant_schedule(optimizer)
#     elif scheduler == 'warmupconstant':
#         return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
#     elif scheduler == 'warmuplinear':
#         return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
#     else:
#         raise ValueError("Unknown scheduler {}".format(scheduler))


# def _get_optimizer(loss_model):
#     weight_decay: float = 0.01 # todo: naar self
#     optimizer_params : Dict[str, object]= {'lr': 5e-5}
#     optimizer_class: Type[Optimizer] = AdamW
#     # Prepare optimizers
#     param_optimizer = list(loss_model.named_parameters())

#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
#         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#     ]
#     # optimizer_params = self.optimizer_params
#     # optimizer = self.optimizer_class(optimizer_grouped_parameters, **optimizer_params)

#     optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

#     return optimizer

            
@dataclass
class FullCfg:
    batch_size: int
    loss_type: str
    debug = False
    head_lr = 5e-5

    factor = 0.8
    epochs = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_encoder_input = 1024
    audio_hidden_dim = 768
    layer_dim = 2

    text_encoder_model = "distilbert-base-uncased"
    
    text_embedding = 768
    mutual_embedding_dim = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0
    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    # final_projection_dim = 768  # [256 or 768]
    audio_dropout = 0.1
    text_dropout = 0.1
    text_pooling: str = 'mean'   #['original', 'cls', 'mean']
    text_model_name : str = 'distilbert-base-uncased'
    text_max_length: int = 32
    

######### This stays in MAIN
def main(args):
    set_logconfig()
    t_start = time()

    # Set training parameters from argparse
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
 
    # scale_type =  args.scale_type
    num_epochs = args.num_epochs # original 1

    # Adjust configurations
    FullCfg.num_epochs = args.num_epochs
    FullCfg.final_projection_dim = args.final_projection_dim
    FullCfg.audio_proj_head = args.audio_proj_head
    FullCfg.text_proj_head = args.text_proj_head
    FullCfg.audio_activation = args.audio_activation
    FullCfg.device = device
    FullCfg.batch_size = args.batch_size
    FullCfg.loss_type = args.loss_type
    FullCfg.text_pooling = args.text_pooling

    # if args.use_lr:
    #     FullCfg.head_lr = 2e-5
    #     FullCfg.audio_encoder_lr = 2e-5
    #     FullCfg.text_encoder_lr = 2e-5
    log_name = setup_config(args, [FullCfg])
    FullCfg.log_name = log_name

    # Setup dataloaders
    data_loader = MMloader(args, FullCfg)
    train_loader = data_loader.train_loader
    test_loader = data_loader.test_loader

    full_model = CLIPModel(FullCfg).to(device)
    # tokenizer = DistilBertTokenizer.from_pretrained(FullCfg.text_tokenizer)
    # print("This is self.tokenizer: ", tokenizer)

    tokenizer = AutoTokenizer.from_pretrained(FullCfg.text_model_name, cache_dir=None)

    # TODO: dit ergens in init dataloader doen
    data_loader.train_dataset.tokenizer = tokenizer
    data_loader.test_dataset.tokenizer = tokenizer
    data_loader.train_dataset.text_max_length = FullCfg.text_max_length
    data_loader.test_dataset.text_max_length = FullCfg.text_max_length

    warmup_steps =  math.ceil(len(train_loader) * num_epochs * 0.1)  # 10% of train data for warm-up
    num_train_steps = int(len(train_loader) * num_epochs)

    FullCfg.eval_every = int(math.ceil(len(train_loader) * 0.1)) #Evaluate every 5% of the data
    FullCfg.print_every = int(math.ceil(len(train_loader) * 0.02)) #Print results every 2% of the data
    print("[main] print_every {} eval_every {} ".format(FullCfg.print_every, FullCfg.eval_every))

    if args.load_model: 
        print("[Main] loading model ", args.load_model_path)
        full_model.load_state_dict(torch.load(args.load_model_path))

    if args.load_checkpoint:
        print("[Main] training from checkpoint ", args.load_checkoint_path)
        checkpoint = torch.load(args.load_model_path)
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        full_model = checkpoint['full_model']
        optimizer = checkpoint['optimizer']
        lr_scheduler = checkpoint['lr_sched']

    else:
        print("[Main] training from scratch ")
        use_old_opt = args.use_old_opt
        use_old_opt = False
        print("[TODO] old opt always set to TRUE")
        if use_old_opt:
            print("[del] us_old_opt ON")
            if full_model.audio_projection and full_model.text_projection:
                params = [
                    {"params": full_model.audio_encoder.parameters(), "lr": FullCfg.audio_encoder_lr},
                    {"params": full_model.text_encoder.parameters(), "lr": FullCfg.text_encoder_lr},
                    {"params": itertools.chain(
                        full_model.audio_projection.parameters(), full_model.text_projection.parameters()
                    ), "lr": FullCfg.head_lr, "weight_decay": FullCfg.weight_decay}
                ]
            elif full_model.audio_projection and not full_model.text_projection:
                params = [
                    {"params": full_model.audio_encoder.parameters(), "lr": FullCfg.audio_encoder_lr},
                    {"params": full_model.text_encoder.parameters(), "lr": FullCfg.text_encoder_lr},
                    {"params": itertools.chain(
                        full_model.audio_projection.parameters()
                    ), "lr": FullCfg.head_lr, "weight_decay": FullCfg.weight_decay}
                ]
            elif not full_model.audio_projection and full_model.text_projection:
                params = [
                    {"params": full_model.audio_encoder.parameters(), "lr": FullCfg.audio_encoder_lr},
                    {"params": full_model.text_encoder.parameters(), "lr": FullCfg.text_encoder_lr},
                    {"params": itertools.chain(
                        full_model.text_projection.parameters()
                    ), "lr": FullCfg.head_lr, "weight_decay": FullCfg.weight_decay}
                ]
            elif not full_model.audio_projection and not full_model.text_projection: # TODO: weight decay over alles?
                params = [
                    {"params": full_model.audio_encoder.parameters(), "lr": FullCfg.audio_encoder_lr},
                    {"params": full_model.text_encoder.parameters(), "lr": FullCfg.text_encoder_lr},
                    {"params": itertools.chain(), "lr": FullCfg.head_lr, "weight_decay": FullCfg.weight_decay}
                ]

            optimizer = torch.optim.AdamW(params, weight_decay=0.)
        else:
            print("[del] us_old_opt OFF")
            # TODO: beter in init van Optimization ++ verget @staticmethod dan niet!
            # optimizer = _get_optimizer(full_model)
            # scheduler: str = 'WarmupLinear'
            # scheduler = _get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", patience=FullCfg.patience, factor=FullCfg.factor
        # )
        epoch = 0

    opt = Optimization(FullCfg, model=full_model)
    opt.train(train_loader, test_loader, epoch)

    t_end = time()
    print("[main] ------------------------------------------------------------")
    print("[main] Done, total duration {} seconds ".format(int(t_end - t_start)))
    csv_file = pd.read_csv(opt.eval_csv_filename)
    print("[main] Maximum text acc step={} acc={}".format(csv_file['text_acc'].idxmax(), csv_file['text_acc'].max()))
    print("[main] Maximum audio acc step={} acc={}".format(csv_file['audio_acc'].idxmax(), csv_file['audio_acc'].max()))
    best_idx = csv_file['mean_acc'].idxmax()
    print("[main] Maximum mean acc step={} acc={}".format(best_idx, csv_file['mean_acc'].max()))
    print(", ".join(["{} - {}".format(k, v) for k, v in csv_file.iloc[best_idx].to_dict().items()]))
    


if __name__ == "__main__":
    # Parse flags in command line arguments
    parser = ArgumentParser()

    parser.add_argument('--train_dataset', default='sp_sample', const='wiki',
                    nargs='?', choices=['wiki', 'sp_sample', 'sp'],
                    help='Name of training dataset (default: %(default)s)')
    parser.add_argument('--test_dataset', default='sp_sample', const='sts',
                    nargs='?', choices=['sts', 'sp_sample',  'sp'],
                    help='Name of test dataset (default: %(default)s)')
    parser.add_argument('--dev_dataset', default='sts', const='sts',
                    nargs='?', choices=['sts'],
                    help='Name of dev dataset (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=100,
                        help='Seet to use')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs to train')
    

    parser.add_argument('--log_dir', type=str, default="./logs",
                        help='Folder where so save logs.')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='batch_size')

    parser.add_argument('--final_projection_dim', type=int, default=256, 
                    nargs='?', choices=[256, 768],
                    help='Final output dimensions of the embeddings')

    # parser.add_argument('--clip_loss', action='store_true')
    parser.add_argument('--loss_type', default='simcse_loss', const='simcse_loss',
                    nargs='?', choices=['simcse_loss', 'clip_loss', 'clip_loss_simple'],
                    help='Name of scale_type (default: %(default)s)')

    # parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--audio_proj_head', default='gru', const='gru',
                    nargs='?', choices=['simple_projection_head', 'rnn', 'gru'],
                    help='Activation to use in simple proj head (default: %(default)s)')
    parser.add_argument('--text_proj_head', default='None', const='None',
                    nargs='?', choices=['simple_projection_head', 'None'],
                    help='Activation to use in simple proj head (default: %(default)s)')
    parser.add_argument('--text_pooling', default='original', const='original',
                    nargs='?', choices=['original', 'cls', 'mean'],
                    help='Pooling method to use for text model (default: %(default)s)')

    parser.add_argument('--audio_activation', default='relu', const='relu',
                    nargs='?', choices=['relu', 'gelu'],
                    help='Activation to use in simple proj head (default: %(default)s)')
    parser.add_argument('--scale_type', default='fixed', const='fixed',
                    nargs='?', choices=['fixed', 'learned'],
                    help='Name of scale_type (default: %(default)s)')
    parser.add_argument('--scale', type=float, default=20,
                        help='Fixed scale to use')

    # GPU device number as in "cuda:0". Defaul is 0.
    parser.add_argument("-cuda", "--cuda_number", type=str, default='0')
    parser.add_argument('--eval_every', type=int, default=100)

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='enables fp16 training')

    parser.add_argument('--use_lr', dest='use_lr', action='store_true',
                        help="Change the learning rate")

    parser.add_argument('--save_model', dest='save_model', action='store_true',
                        help="Save the model weights.")
    parser.add_argument('--load_model_path', type=str, default="./logs/load_test/output/full_model_weights.pth",
                        help='Folder where model weights are saved.')
    parser.add_argument('--load_model', dest='load_model', action='store_true',
                        help="Load the model in model_path to continue a downstream task")

    parser.add_argument('--save_checkpoint', dest='save_checkpoint', action='store_true',
                        help="Save the model, optimizer and scheduler weights.")
    parser.add_argument('--load_checkpoint_path', type=str, default="./logs/load_test/output/checkpoint.pth",
                        help='Folder where so save logs.')
    parser.add_argument('--load_checkpoint', dest='load_checkpoint', action='store_true',
                        help="Load a model, optimizer and scheduler to continue training.")

    parser.add_argument('--use_old_opt', dest='use_old_opt', action='store_true',
                        help="test: use old or new opt.")
    args, unparsed = parser.parse_known_args()

    main(args)