"""
Contrastive multimodal learning
Author: Casper Wortmann
Usage: python main.py
"""
import logging
from argparse import ArgumentParser

import time
from time import time
from datetime import datetime
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
# from transformers import AdamW 
# import torch.nn as nn, Tensor 
from torch import nn, Tensor
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

# Load all static config stuff
conf = OmegaConf.load("./config.yaml")
print("[cudacheck] Is cuda available? ", torch.cuda.is_available())

################
# move to Utils
def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def set_logconfig():
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.INFO)

def get_log_name(args):

    log_name = "mm-{}_{}_{}_{}_{}".format(args.loss_type, args.proj_head, 
            args.proj_head_type, args.scale_type, 
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_name = os.path.join(args.log_dir, log_name)
    return log_name

def setup_config(args, dataclasses):
    # Set the seed and create output directories
    set_seed(args)
    log_name = get_log_name(args)
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


class DevInputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', texts: List[str] = None, audio_embeds: List[str] = None, label: Union[int, float] = 0):
        """
        Creates one InputExample with the given texts, guid and label
        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param audio_embeds
            the audio embedings for the example.
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.audio_embeds = audio_embeds
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, texts: {}, audio_embeds: {}".format(str(self.label), "; ".join(self.texts), str(self.audio_embeds))

class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', texts: List[str] = None, mean_embeds: List[str] = None, full_embeds: List[str] = None):
        """
        Creates one InputExample with the given texts, guid and label
        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param audio_embeds
            the audio embedings for the example.
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.mean_embeds = mean_embeds
        self.full_embeds = full_embeds

    def __str__(self):
        return "<InputExample> texts: {}, mean_embeds: {}, full_embeds: {}".format("; ".join(self.texts), str(self.mean_embeds), str(self.full_embeds[0].shape))

################
# move to dataloaders.py
class MMloader(object):
    """
    Loads one of the supported datasets.
    Supported datasets:
        sts dataset
        sample data (scraped)
        SP dataset (future)
    """
    def __init__(self, args, kwargs={}):

        train_dataset_name = args.train_dataset
        dev_dataset_name = args.dev_dataset
        test_dataset_name = args.test_dataset

        train_batch_size = args.train_batch_size
        self.train_batch_size = train_batch_size

        self.device = None # useless line, delete

        if args.proj_head == 'simple_gru':
            self.load_full = True
        else:
            self.load_full = False

        print("[MMloader] train dataset ", train_dataset_name)

        # Get the datasets
        if train_dataset_name == "wiki":
            train_dataset = self.get_wiki_dataset()
            self.train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, drop_last=True, **kwargs)
        elif train_dataset_name == "sp_sample":
            train_dataset = self.get_sp_dataset(directory=conf.sp_sample_path, traintest="train", load_full=self.load_full, device=self.device)
            self.train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, drop_last=True, **kwargs)
        elif train_dataset_name == "sp":
            train_dataset = self.get_sp_dataset(directory=conf.sp_path,  traintest="train", load_full=self.load_full, device=self.device)
            self.train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True, **kwargs)
        else:
            raise Exception('Unknown dataset')
        
        self.train_dataset = train_dataset
        print("[MMloader] train dataset loaded, length: ", len(train_dataset))

        # if dev_dataset_name == "sts":
        #     sts_dataset = self.get_sts_dataset()
        #     dev_samples = sts_dataset.dev_samples
        #     #self.dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')
        # else:
        #     raise Exception('Unknown dataset')
        # print("[MMloader] dev dataset loaded, length: ", len(dev_samples))

        if test_dataset_name == "sts":
            sts_dataset = self.get_sts_dataset()
            test_samples = sts_dataset.test_samples
            #self.test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
        elif test_dataset_name == "sp_sample":
            # test_dataset = self.get_sp_sample_dataset(train_set=False, directory=conf.sp_sample_path, load_full=self.load_full)
            test_dataset = self.get_sp_dataset(directory=conf.sp_sample_path,  traintest="test", load_full=self.load_full, device=self.device)
            self.test_loader = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False, drop_last=True, **kwargs)
        elif test_dataset_name == "sp":
            # test_dataset = self.get_sp_sample_dataset(train_set=False, directory=conf.sp_path, load_full=self.load_full)
            test_dataset = self.get_sp_dataset(directory=conf.sp_path,  traintest="test",  load_full=self.load_full, device=self.device)
            self.test_loader = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False, drop_last=True, **kwargs)
            # self.test_evaluator = EmbeddingMatrixEvaluator.EmbeddingMatrixEvaluator.from_input_dataloader(test_loader, batch_size=train_batch_size, name='sp-test')

        else:
            raise Exception('Unknown dataset')
        self.test_dataset = test_dataset
        print("[MMloader] test dataset loaded, length: ", len(test_dataset))


    def get_wiki_dataset(self):
        dataset =  wikiDataset()
        # train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)
        return dataset

    def get_sts_dataset(self):
        dataset = stsDataset()
        return dataset

    # def get_sp_sample_dataset(self, train_set=True, directory=conf.sp_sample_path, load_full=False):
    #     dataset =  spDatasetNoMemory(train_set=train_set, directory=directory, load_full=load_full)
    #     return dataset

    def get_sp_dataset(self, directory=conf.sp_sample_path, traintest="train", load_full=False, device=None):
        dataset =  spDatasetNoMemory(directory=directory, traintest=traintest,  load_full=load_full, device=device)
        return dataset

class wikiDataset(datautil.Dataset):
    """
    Wiki dataset dataloader. 
    """
    def __init__(self, directory='../data/wiki1m_for_simcse.txt'):
        print("[wikiDataset]")
        # We use 1 Million sentences from Wikipedia to train our model
        wikipedia_dataset_path = directory
        if not os.path.exists(wikipedia_dataset_path):
            print("[wikiDataset] wiki1m_for_simcse does not exists - start downloading")
            util.http_get('https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt', wikipedia_dataset_path)

        # train_samples is a list of InputExample objects where we pass the same sentence twice to texts, i.e. texts=[sent, sent]
        train_samples = []
        with open(wikipedia_dataset_path, 'r', encoding='utf8') as fIn:
            for line in fIn:
                line = line.strip()
                if len(line) >= 10:
                    train_samples.append(InputExample(texts=[line, line]))

        self.train_samples = train_samples
        print("[wikiDataset] len of train samples: ", len(train_samples))

    def __len__(self):
        """ Denotes the total number of utterances """
        return len(self.train_samples)

    def __getitem__(self, index):
        """ Return one item from the df """
        sample = self.train_samples[index]
        return sample


class stsDataset(datautil.Dataset):
    """
    Sts dataset dataloader. 
    """
    def __init__(self, directory='../data/stsbenchmark.tsv.gz'):
        print("[stsDataset]")
        # Check if dataset exsist. If not, download and extract  it
        sts_dataset_path = directory
        if not os.path.exists(sts_dataset_path):
            print("[stsDataset] sts benchmark does not exists - start downloading")
            util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

        
        # Read STSbenchmark dataset and use it as development set
        logging.info("Read STSbenchmark dev dataset")
        dev_samples = []
        test_samples = []
        with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1

                if row['split'] == 'dev':
                    dev_samples.append(DevInputExample(texts=[row['sentence1'], row['sentence2']], label=score))
                elif row['split'] == 'test':
                    test_samples.append(DevInputExample(texts=[row['sentence1'], row['sentence2']], label=score))

        self.dev_samples = dev_samples
        self.test_samples = test_samples

    def __len__(self):
        """ Denotes the total number of utterances """
        return len(self.dev_samples), len(self.test_samples)

    def __getitem__(self, index):
        """ Return one item from the df """
        print("getitem for sts?")
        raise NotImplementedError

class spDataset_old(datautil.Dataset):
    """
    Spotify Podcast dataset dataloader. 
    """
    def __init__(self, train_set=True, directory=conf.sp_sample_path, load_full=False):
        print("[spDataset] init from directory ", directory)

        h5py_files = list(Path(directory).glob('*.h5'))
        print("[spDataset] found {} h5py files".format(len(h5py_files)))
        samples = []
        self.max_embed_dim = 0
        for h5idx, h5py_file in enumerate(h5py_files):
            f = h5py.File(h5py_file, 'r')
            for idx, (_, train_split, sent, mean_embeds) in enumerate((zip(f['filenames'], f['train_split'], f['sentences'], f['mean_embeddings']))):
                # if idx % 5000 == 0 and idx > 0:
                #     print("[spdataset] loaded ", idx)
                if train_split == train_set:
                    if load_full:
                        full_embeds = torch.Tensor(np.array(f[str(idx)]))
                        # samples.append(InputExample(texts=[sent.decode("utf-8")], mean_embeds=None, full_embeds=[full_embeds]))
                        samples.append(InputExample(texts=[sent], mean_embeds=None, full_embeds=[full_embeds]))
                    else:
                        # samples.append(InputExample(texts=[sent.decode("utf-8")], mean_embeds=[torch.Tensor(mean_embeds)], full_embeds=None))
                        samples.append(InputExample(texts=[sent], mean_embeds=[torch.Tensor(mean_embeds)], full_embeds=None))

            self.max_embed_dim = max(self.max_embed_dim, f.attrs['max_embed_dim'])
            f.close()
            if h5idx % 10 == 0 and h5idx > 0:
                #     print("[spdataset] loaded ", idx)
                print("[spdataset] loaded {}/{}".format(h5idx, len(h5py_files)))

        self.samples = samples

    def __len__(self):
        """ Denotes the total number of utterances """
        return len(self.samples)

    def __getitem__(self, index):
        """ Return one item from the df """
        sample = self.samples[index]
        return sample

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
            f = h5py.File(h5py_file, 'r')

            for sentidx in range(len(f['sentences'])):
                idx2file[sample_idx] = (h5idx, sentidx)
                sample_idx += 1
 
            self.max_embed_dim = max(self.max_embed_dim, f.attrs['max_embed_dim'])
            f.close()
            if h5idx % 10 == 0 and h5idx > 0:
                print("[spdataset] loaded {}/{}".format(h5idx, len(h5py_files)))
        self.idx2file = idx2file

        # if self.load_full:
        #     self.collate_fn = self.full_batching_collate
        # else:
        #     self.collate_fn = self.mean_batching_collate

    def __len__(self):
        """ Denotes the total number of utterances """
        return len(self.idx2file.keys())

    def __getitem__(self, index):
        """ Return one item from the df """

        # if load_full:
        #                 full_embeds = torch.Tensor(np.array(f[str(idx)]))
        #                 samples.append(InputExample(texts=[sent.decode("utf-8")], mean_embeds=None, full_embeds=[full_embeds]))
        #             else:
        #                 samples.append(InputExample(texts=[sent.decode("utf-8")], mean_embeds=[torch.Tensor(mean_embeds)], full_embeds=None))

        if self.load_full:
            h5py_idx, sent_idx = self.idx2file[index]
            h5py_file = self.h5py_idx2file[h5py_idx]

            f = h5py.File(h5py_file, 'r')
            # sent = f['sentences'][sent_idx].decode("utf-8")
            sent = f['sentences'][sent_idx]
            full_embeds = torch.Tensor(np.array(f[str(sent_idx)]))
            # sample = (sent, full_embeds)
            sample = InputExample(texts=[sent], mean_embeds=None, full_embeds=[full_embeds])
            f.close()

        
        else:
            h5py_idx, sent_idx = self.idx2file[index]
            h5py_file = self.h5py_idx2file[h5py_idx]

            f = h5py.File(h5py_file, 'r')

            # sent = f['sentences'][sent_idx].decode("utf-8")
            sent = f['sentences'][sent_idx]
            mean_embeds = torch.Tensor(f['mean_embeddings'][sent_idx])
            # sample = (sent, mean_embeds)
            sample = InputExample(texts=[sent], mean_embeds=mean_embeds, full_embeds=None)
            f.close()
            
        return sample


######################################################
# MODULES
######################################################

class simple_projection_head(nn.Module):
    """
    :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
    """
    def __init__(self, input_dim, hidden_dim=768, output_dim=768, 
        proj_type = 'gelu'):
        super(simple_projection_head, self).__init__()

        self.config_keys = ['input_dim', 'hidden_dim', 'output_dim']
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        if proj_type == 'gelu':
            self.simple_model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Dropout(),
                nn.LayerNorm(output_dim),
            )
        elif proj_type == 'relu':
            self.simple_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim), # TODO: experiment with batchnormalization
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            #nn.ReLU()
        )


    def __repr__(self):
        return "Simple_projection_head({}) ".format(self.get_config_dict())

    def forward(self, audio_seqlen):
        """Returns token_embeddings, cls_token"""
        features, _ = audio_seqlen
        features = self.simple_model(features)

        return features

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        # TODO
        raise NotImplementedError
    @staticmethod
    def load(input_path: str):
        # TODO
        raise NotImplementedError


class simple_gru(nn.Module):
    """
    :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
    """
    def __init__(self,  input_dim, hidden_dim, num_layers, output_dim=768, dropout_prob=0.1, 
                use_softmax = False, pad_pack = True, device=None):
        super(simple_gru, self).__init__()

        self.config_keys = ['input_dim', 'hidden_dim', 'num_layers', 'output_dim', 'dropout_prob']
        
        # Defining the number of layers and the nodes in each layer
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.use_softmax = use_softmax

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        if use_softmax:
            self.softmax = nn.Softmax(dim=1)
            
        # self.pad_pack = pad_pack
        self.pad_pack = False
        self._target_device = device

    def __repr__(self):
        return "simple_gru({}) ".format(self.get_config_dict())

    def forward(self, audio_seq):
        """Returns token_embeddings, cls_token"""

        features, seq_lens = audio_seq
        seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=torch.device('cpu'))

        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.num_layers, features.size(0), self.hidden_dim).requires_grad_()
        
        if self.pad_pack:
            print("PACK PAD ON!!")
            # Pack the features such that we do not compute zero products
            features = pack_padded_sequence(features, seq_lens, batch_first=True, enforce_sorted=False)

        # TODO: fix a check fo this on cpu
        # if torch.cuda.is_available():
        #     h0 = h0.cuda()
        h0 = h0.to(self._target_device)

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(features, h0.detach())

        if self.pad_pack:
            # undo the packing operation
            out, _ = pad_packed_sequence(out, batch_first=True)

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        
        if self.use_softmax:
            out = self.softmax(out)

        return out

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        # TODO
        raise NotImplementedError
                 
    @staticmethod
    def load(input_path: str):
        # TODO
        raise NotImplementedError
# model = simple_lstm(input_dim=1024, hidden_dim=768, num_layers=2, output_dim=768, dropout_prob=0.1, use_softmax = True)
              

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

@dataclass
class mmAudioCfg:
    proj_head: str = "simple_projection_head"
    proj_type: str = 'relu'
    input_dim: int = 1024
    hidden_dim: int = 768
    output_dim: int = 768
    pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')



@dataclass
class mmTextCfg:
    proj_head: str = 'distilbert-base-uncased'
    text_model_name: str = 'distilbert-base-uncased'
    # max_length: int = 200
    add_pooling: bool = False
    pooling_mode : str = 'cls'



class textModule(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Adjusted for a multimodal setup
    Loads the correct class, e.g. BERT / RoBERTa etc.
    :param model_name_or_path: Huggingface models name (https://huggingface.co/models) etc
    """
    def __init__(self, t_cfg=mmTextCfg, 
                # model_name_or_path: str, 
                max_seq_length: Optional[int] = None,
                model_args: Dict = {}, cache_dir: Optional[str] = None,
                tokenizer_args: Dict = {}, do_lower_case: bool = False,
                tokenizer_name_or_path : str = None):
        super(textModule, self).__init__()

        model_name_or_path = t_cfg.text_model_name
        self.model_name = t_cfg.text_model_name
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        my_dropout = 0.1
        config_kwargs = {
            "cache_dir": cache_dir,
            "attention_dropout": my_dropout,
            "dropout": my_dropout,
        }

        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, **config_kwargs)
        self.hidden_dim = config.dim
        print("[Transformer_multimodal] Configurations used: \n", config)

        self._load_model(model_name_or_path, config, cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path, cache_dir=cache_dir, **tokenizer_args)

        # #No max_seq_length set. Try to infer from model
        # if max_seq_length is None:
        #     if hasattr(self.auto_model, "config") and hasattr(self.auto_model.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
        #         max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)
        #         print("INFERRED FROM MODEL ")

        self.max_seq_length = max_seq_length

        # print("Setting max seq len to : ", self.max_seq_length)
        
        # if tokenizer_name_or_path is not None:
        #     print("And there is this tokenizer thing again")
        #     self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__


    def _load_model(self, model_name_or_path, config, cache_dir):
        """Loads the transformer model"""
        if isinstance(config, T5Config):
            self._load_t5_model(model_name_or_path, config, cache_dir)
        else:
            self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

    def _load_t5_model(self, model_name_or_path, config, cache_dir):
        """Loads the encoder model from T5"""
        from transformers import T5EncoderModel
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = T5EncoderModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

    def __repr__(self):
        return "Transformer({}) with Transformer model: {} ".format(self.get_config_dict(), self.auto_model.__class__.__name__)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""

        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        # if 'token_type_ids' in features:
        #     print("token type ids inii t!!")
        #     trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        ### NOte: in mijn func gaat output_states hier al naar pooling

        # Original was like this
        output_tokens = output_states[0]
        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        to_tokenize = [texts]
        # if isinstance(texts[0], str):
        #     to_tokenize = [texts]
        # elif isinstance(texts[0], dict):
        #     to_tokenize = []
        #     output['text_keys'] = []
        #     for lookup in texts:
        #         text_key, text = next(iter(lookup.items()))
        #         to_tokenize.append(text)
        #         output['text_keys'].append(text_key)
        #     to_tokenize = [to_tokenize]
        # else:
        #     batch1, batch2 = [], []
        #     for text_tuple in texts:
        #         batch1.append(text_tuple[0])
        #         batch2.append(text_tuple[1])
        #     to_tokenize = [batch1, batch2]

        #strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length))
        return output

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return Transformer(model_name_or_path=input_path, **config)


class audioModule(nn.Module):
    """
        Audiomodule
    """
    def __init__(self, a_cfg=mmAudioCfg, ):
        super(audioModule, self).__init__()
        self.input_dim = a_cfg.input_dim
        self.hidden_dim = a_cfg.hidden_dim
        self.output_dim = a_cfg.output_dim
        self.proj_type = a_cfg.proj_type
        self.config_keys = ['input_dim', 'hidden_dim', 'output_dim', 'proj_type']

        if a_cfg.proj_head == "simple_projection_head":
            self.audio_projection = simple_projection_head(a_cfg.input_dim, a_cfg.hidden_dim, a_cfg.output_dim, proj_type=a_cfg.proj_type)
            self.features_needed = 'mean_audio'
        elif a_cfg.proj_head == "simple_gru":
            self.audio_projection = simple_gru(a_cfg.input_dim, a_cfg.hidden_dim, 
                    num_layers=2, output_dim=a_cfg.output_dim, 
                    dropout_prob=0.1, use_softmax = True, pad_pack=False, device=a_cfg.device)

            self.features_needed = 'full_audio'

    def forward(self, audio_seq):
        return self.audio_projection(audio_seq)

    def _load_model(self, model_name_or_path, config, cache_dir):
        """Loads the transformer model"""
        raise NotImplementedError
        if isinstance(config, T5Config):
            self._load_t5_model(model_name_or_path, config, cache_dir)
        else:
            self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

    def _load_t5_model(self, model_name_or_path, config, cache_dir):
        """Loads the encoder model from T5"""
        raise NotImplementedError

    def __repr__(self):
        return "Audiomodule({})".format(self.get_config_dict())

    def get_embedding_dimension(self) -> int:
        return self.output_dim

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        raise NotImplementedError
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        raise NotImplementedError
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break
        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return Transformer(model_name_or_path=input_path, **config)


class mmModule(nn.Module): 
    def __init__(
            self,
            text_modules,
            audio_modules,
            batch_size=128,
            device: torch.device = torch.device('cpu'),
    ):
        super().__init__()

        # Set the batching function
        self.features_needed = audio_modules[0].features_needed
        if self.features_needed == 'mean_audio':
            self._batching_fn = self.mean_batching_collate
        else:
            self._batching_fn = self.full_batching_collate

        # Create the full model
        if text_modules is not None and not isinstance(text_modules, OrderedDict):
            text_modules = OrderedDict([(str(idx), module) for idx, module in enumerate(text_modules)])
        if audio_modules is not None and not isinstance(audio_modules, OrderedDict):
            audio_modules = OrderedDict([(str(idx), module) for idx, module in enumerate(audio_modules)])

        self.text_model = nn.Sequential(text_modules)
        self.audio_model = nn.Sequential(audio_modules)
        self._target_device = device
        self.batch_size = batch_size

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes the texts
        """
        tokenized = self.text_model._modules[next(iter(self.text_model._modules))].tokenize(texts)

        return tokenized

    def mean_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        texts = []
        mean_embeds = []

        for example in batch:
            texts.append(example.texts)
            mean_embeds.append(example.mean_embeds)

        # Send audio features to target devce
        audio_features = torch.stack(mean_embeds).to(self._target_device)

        # Tokenize sentences and send to tarket device
        tokenized = self.tokenize(texts)
        sentence_features = []
        for key in tokenized:
            if isinstance(tokenized[key], torch.Tensor):
                tokenized[key] = tokenized[key].to(self._target_device)
        sentence_features.append(tokenized)

        return sentence_features, audio_features, None

    def full_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        texts = []
        mean_embeds = []
        full_embeds = []
        padded_full_embeds = []
        seq_lens = []

        for example in batch:
            texts.extend(example.texts)
            # mean_embeds.extend(example.mean_embeds)
            full_embeds.append(example.full_embeds[0])

        # Send audio features to target devce
        # mean_embeds = torch.stack(mean_embeds).to(self._target_device)

        seq_lens = [em.shape[0] for em in full_embeds]
        padded_full_embeds = pad_sequence(full_embeds, batch_first=True).to(self._target_device)

        # Tokenize sentences and send to target device
        tokenized = self.tokenize(texts)
        sentence_features = []
        for key in tokenized:
            if isinstance(tokenized[key], torch.Tensor):
                tokenized[key] = tokenized[key].to(self._target_device)
        sentence_features.append(tokenized)

        return sentence_features, padded_full_embeds, seq_lens

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
        train_dataloader: DataLoader,
        loss_model: nn.Module,
        dev_evaluator = None,
        test_dataloader = None,
        epochs: int = 1,
        steps_per_epoch = None,
        scheduler: str = 'WarmupLinear',
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 2,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0):

        self.use_amp = use_amp
        self.weight_decay = weight_decay
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        self.text_model.to(self._target_device)
        self.audio_model.to(self._target_device)
        loss_model.to(self._target_device)

        # Set the collate function for the dataloader       # TODO: init van matrix_check hierbij zetten
        # self.batch_size = train_dataloader.train_batch_size
        # self.train_dataloader = train_dataloader
        train_dataloader.collate_fn = self._batching_fn

        steps_per_epoch = len(train_dataloader)
        num_train_steps = int(steps_per_epoch * epochs)
        print("[del] ", steps_per_epoch, num_train_steps, warmup_steps)

        optimizer = self._get_optimizer(loss_model)
        scheduler = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        skip_scheduler = False
        losses = []
        training_steps = 0
        running_loss, running_acc_a, running_acc_t = 0, 0, 0
        loss_freq = args.eval_every

        if loss_model.scale_type != 'learned':
            update_scale = False
        else:
            update_scale = True

        print("[fit] Performance before training: ")
        # TODO: uncomment
        self.perform_matrix_evaluation(loss_model, -1, -1)

        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            t1 = time()
            loss_model.zero_grad()
            loss_model.train()

            for new_train_index, (sent_features, audio_features, seq_len) in enumerate(train_dataloader):
                loss_value, metrics = loss_model(sent_features, audio_features, seq_len)
                running_loss += loss_value.item()

                if new_train_index > 0 and training_steps % loss_freq == 0:
                    mean_loss = running_loss / loss_freq
                    # print("\t\t\t[del] mean loss: ", mean_loss)
                    self.add_train_logging(epoch, training_steps, mean_loss, metrics)
                    losses.append(mean_loss)
                    to_plot(self.train_csv_filename, column='loss', title="Train loss")
                    running_loss = 0

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                training_steps += 1

                if not skip_scheduler:
                    scheduler.step()

                # check this
                # if update_scale:
                #     with torch.no_grad():
                #         loss_model.logit_scale.data.clamp_(-np.log(100), np.log(100))


                if new_train_index > 0 and new_train_index % evaluation_steps == 0:
                    # self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback) #TODO: add sts
                    self.perform_matrix_evaluation(loss_model, epoch, training_steps)
                    loss_model.zero_grad()
                    loss_model.train()
                    to_plot(self.eval_csv_filename, column='mean_acc', title="Test accuracy (mean)")

                # # [del] check on one batch
                # print("[metrics] ", loss_value.item(), metrics['mean_acc'])
                # if new_train_index >= 0:
                #     print("[del] break")
                #     break            
            # Plot loss and accuracy curves
            t2 = time()
            print("[train] epoch duration {} seconds".format(int(t2-t1)))
        

    def init_logging(self):
        self.model_save_path = '{}/output'.format(self.log_name)
        os.makedirs(self.model_save_path, exist_ok=True)
        self.train_csv_filename = os.path.join(self.model_save_path, "train.csv")
        self.eval_csv_filename = os.path.join(self.model_save_path, "test.csv")

    def add_train_logging(self, epoch, steps, loss, metrics):
        output_file_exists = os.path.isfile(self.train_csv_filename)

        if not output_file_exists:
            self.metric_headers = [metric for metric in metrics.keys()]
            self.train_csv_headers = ["epoch", "steps", "loss"] + self.metric_headers
            with open(self.train_csv_filename , newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.train_csv_headers)

        # Add metrics to CSV
        metric_vals = [metrics[header] for header in self.metric_headers]
        with open(self.train_csv_filename , newline='', mode="a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, steps, loss] + metric_vals)

    def add_logging(self, epoch, steps, metrics):
        output_file_exists = os.path.isfile(self.eval_csv_filename)
        if not output_file_exists:
            self.metric_headers = [metric for metric in metrics.keys()]
            self.eval_csv_headers = ["epoch", "steps"] + self.metric_headers
            with open(self.eval_csv_filename , newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.eval_csv_headers)

        # Add metrics to CSV
        metric_vals = [metrics[header] for header in self.metric_headers]
        with open(self.eval_csv_filename , newline='', mode="a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, steps] + metric_vals)

    def setup_matrix_evaluation(self, dataloader, 
            log_name: str = '', 
            show_progress_bar: bool = False, write_csv: bool = True):

        # TODO: dit kan allemaal naar INIT
        # dataloader.collate_fn  = self._batching_fn
        self.eval_dataloader = dataloader

        self.eval_dataloader.collate_fn  = self._batching_fn

        self.write_csv = write_csv

        self.main_similarity = dek_cos_sim
        self.log_name = log_name
        self.show_eval_progress_bar = show_progress_bar
        self.init_logging()

    def perform_matrix_evaluation(self, train_loss_func, epoch: int = -1, steps: int = -1):
        mean_metrics = self.eval_encode(train_loss_func, show_progress_bar=self.show_eval_progress_bar, convert_to_numpy=True)
        self.add_logging(epoch, steps, mean_metrics)
        print("[Test] Epoch {} step {},\t Acc_audio {:.4f},\t Acc_text  {:.4f}".format(epoch, steps, mean_metrics['audio_acc'], mean_metrics['text_acc']))
        
    def eval_encode(self, loss_model,
            batch_size: int = 32,
            show_progress_bar: bool = None,
            output_value: str = 'sentence_embedding',
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = None,
            normalize_embeddings: bool = False):

        # Set evaluation mode on
        self.eval()

        # Set variables correctly
        if convert_to_tensor:
            convert_to_numpy = False
        if output_value != 'sentence_embedding':
            convert_to_tensor = False
            convert_to_numpy = False
        if device is None:
            device = self._target_device

        # Iterate over data and calculate accuracies
        self.to(device)
        total_len = len(self.eval_dataloader)

        full_validation = False
        if full_validation:
            for idx, batch in enumerate(iter(self.eval_dataloader)):
                sent_features, audio_features, seq_len = batch

                if self.use_amp:
                    print("Adjested by Casp, not yet fully implemented")
                    raise NotImplementedError
                else:
                    # loss_value = loss_model(sent_features, audio_features, labels)
                    with torch.no_grad():
                        loss_value, metrics = loss_model(sent_features, audio_features, seq_len)
                        if idx == 0:
                            #metrics_sum = metrics.copy()
                            met_sum = Counter(metrics.copy())
                        else:
                            #metrics_sum = {k: metrics_sum.get(k, 0) + metrics.get(k, 0) for k in set(metrics_sum)}
                            met_sum.update(Counter(metrics))
        else:
            total_len = 100
            iterator = iter(self.eval_dataloader)
            for idx in range(total_len):
                batch = next(iterator)
                sent_features, audio_features, seq_len = batch

                if self.use_amp:
                    print("Adjested by Casp, not yet fully implemented")
                    raise NotImplementedError
                else:
                    # loss_value = loss_model(sent_features, audio_features, labels)
                    with torch.no_grad():
                        loss_value, metrics = loss_model(sent_features, audio_features, seq_len)
                        if idx == 0:
                            #metrics_sum = metrics.copy()
                            met_sum = Counter(metrics.copy())
                        else:
                            #metrics_sum = {k: metrics_sum.get(k, 0) + metrics.get(k, 0) for k in set(metrics_sum)}
                            met_sum.update(Counter(metrics))

        mean_metrics = {k: value / total_len  for k, value in met_sum.items()}
        del met_sum
        return mean_metrics


class multimodal_loss(nn.Module):
    """
        This loss expects as input a batch consisting of ... etc
    """
    def __init__(self, full_model, scale: float = 20.0, device=None, loss_type='clip_loss', normalize=False, scale_type="fixed"):
        """
        test
        """
        super(multimodal_loss, self).__init__()

        self.text_model = full_model.text_model
        self.audio_model = full_model.audio_model

        self.normalize = normalize
        self.loss_type = loss_type
        self.scale_type = scale_type
        if self.scale_type == 'fixed':
            self.fixed_scale = scale
        elif self.scale_type == 'learned':
            # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            # self.init_parameters_logtiscale()

            self.logit_scale = nn.Parameter(torch.log(torch.ones([]) * 100))
            self.logit_scale.requires_grad = True

        self.similarity_fct = dek_cos_sim
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self._target_device = torch.device(device)
        self.batch_size = full_model.batch_size

    def init_parameters_logtiscale(self):
        # nn.init.normal_(self.token_embedding.weight, std=0.02)
        # nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], audio_features: Tensor, seq_len):
        # Get sentence Representations (shape [batchsize, 768])
        reps_sentences = self.text_model(sentence_features[0])['sentence_embedding']
        # print("[clipmodel] text_embeddings: ", reps_sentences[0])
        
        # Get Audio representations
        reps_audio = self.audio_model((audio_features, seq_len))
        # print("[clipmodel] audio_embeddings: ", reps_audio[0])
        # print("------------------")

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
                audio_logits = audio_logits / audio_logits.norm(dim=1, keepdim=True)
                text_logits = text_logits / text_logits.norm(dim=1, keepdim=True)

            # ground_truth = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  
            ground_truth = torch.arange(self.batch_size, dtype=torch.long, device=self._target_device)
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

            texts_loss = del_cross_entropy(logits, ground_truth, reduction='none')
            images_loss = del_cross_entropy(logits.T, ground_truth.T, reduction='none')
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
        return {'scale_type': self.scale_type, 'similarity_fct': self.similarity_fct.__name__}


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
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        #metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
    metrics['mean_acc'] = np.mean(accs)


    return metrics


### delete
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


def del_cross_entropy(preds, targets, reduction='none'):
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

######### This stays in MAIN
def main(args):
    set_logconfig()
    t_start = time()

    # Set training parameters from argparse
    support_FP16 = args.fp16  #Set to True, if your GPU supports FP16 cores
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    loss_type = args.loss_type
    normalize = args.normalize
    scale_type =  args.scale_type
    num_epochs = args.num_epochs # original 1
    max_seq_length = 32 #todo: 512? # og 32

    # Adjust configurations
    if args.proj_head == 'simple_gru':
        mmAudioCfg.proj_head = 'simple_gru'
    proj_head_type = args.proj_head_type
    if proj_head_type == 'gelu':
        mmAudioCfg.proj_type = 'gelu'
    mmAudioCfg.device = device
    log_name = setup_config(args, [mmAudioCfg, mmTextCfg])

    # Setup dataloaders
    data_loader = MMloader(args)
    train_dataloader = data_loader.train_loader
    test_dataloader = data_loader.test_loader

    # dev_evaluator = data_loader.dev_evaluator      #TODO: STS CHECK
    # This is moved to later stadium TODO: fix that it would also work if sts
    # test_evaluator = data_loader.test_evaluator

    text_embedding_model = textModule(t_cfg=mmTextCfg, max_seq_length=max_seq_length)
    pooling_model = Pooling(text_embedding_model.get_word_embedding_dimension())

    audio_model = audioModule(a_cfg=mmAudioCfg)

    full_model = mmModule(text_modules=[text_embedding_model, pooling_model], 
            audio_modules=[audio_model], 
            batch_size=args.train_batch_size,
            device=device)
    full_model.setup_matrix_evaluation(data_loader.test_loader, log_name=log_name)

    print("[main] model setup complete, model: \n", full_model)
    train_loss = multimodal_loss(full_model, scale=args.scale, device=device, loss_type=loss_type, normalize=normalize, scale_type=scale_type)
    warmup_steps =  math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    evaluation_steps = int(math.ceil(len(train_dataloader) * 0.1)) #Evaluate every 10% of the data

    full_model.fit(
        train_dataloader=train_dataloader,
        loss_model=train_loss,
        dev_evaluator=None,
        test_dataloader = test_dataloader, # Dit kan gebruikt worden om setup amtrix evaluation te doen
        epochs=num_epochs,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=None,
        save_best_model= True,
        optimizer_class=AdamW,
        optimizer_params={'lr': 5e-5},
        use_amp=support_FP16,        #Set to True, if your GPU supports FP16 cores
        show_progress_bar=False
    )
   
    to_plot(full_model.train_csv_filename, column='audio_acc', title="Train accuracy (audio)")
    to_plot(full_model.train_csv_filename, column='text_acc', title="Train accuracy (text)")
    to_plot(full_model.train_csv_filename, column='loss', title="Train loss")

    to_plot(full_model.eval_csv_filename, column='mean_acc', title="Test accuracy (mean)")
    to_plot(full_model.eval_csv_filename, column='audio_acc', title="Test accuracy (audio)")
    to_plot(full_model.eval_csv_filename, column='text_acc', title="Test accuracy (text)")

    to_plot(full_model.eval_csv_filename, column='text_R@10', title="Test R@10 (text)")
    to_plot(full_model.eval_csv_filename, column='audio_R@10', title="Test R@10 (audio)")


    csv_file = pd.read_csv(full_model.eval_csv_filename)
    print("[main] ------------------------------------------------------------")
    print("[main] Maximum text acc step={} acc={}".format(csv_file['text_acc'].idxmax(), csv_file['text_acc'].max()))
    print("[main] Maximum audio acc step={} acc={}".format(csv_file['audio_acc'].idxmax(), csv_file['audio_acc'].max()))

    best_idx = csv_file['mean_acc'].idxmax()
    print("[main] Maximum mean acc step={} acc={}".format(best_idx, csv_file['mean_acc'].max()))
    # print("[main] Results from this: ", csv_file.loc[[best_idx]])
    print(", ".join(["{} - {}".format(k, v) for k, v in csv_file.iloc[best_idx].to_dict().items()]))
    t_end = time()
    print("[main] Done, total duration {} seconds ".format(int(t_end - t_start)))

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
    parser.add_argument('--scale', type=float, default=20,
                        help='Fixed scale to use')

    parser.add_argument('--log_dir', type=str, default="./logs",
                        help='Folder where so save logs.')
    parser.add_argument('--train_batch_size', type=int, default=128, 
                        help='batch_size')

    # parser.add_argument('--clip_loss', action='store_true')
    parser.add_argument('--loss_type', default='simcse_loss', const='simcse_loss',
                    nargs='?', choices=['simcse_loss', 'clip_loss', 'clip_loss_simple'],
                    help='Name of scale_type (default: %(default)s)')

    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--proj_head', default='simple_gru', const='simple_gru',
                    nargs='?', choices=['simple_projection_head', 'simple_gru'],
                    help='Activation to use in simple proj head (default: %(default)s)')
    parser.add_argument('--proj_head_type', default='relu', const='relu',
                    nargs='?', choices=['relu', 'gelu'],
                    help='Activation to use in simple proj head (default: %(default)s)')
    parser.add_argument('--scale_type', default='fixed', const='fixed',
                    nargs='?', choices=['fixed', 'learned'],
                    help='Name of scale_type (default: %(default)s)')

    # GPU device number as in "cuda:0". Defaul is 0.
    parser.add_argument("-cuda", "--cuda_number", type=str, default='0')
    parser.add_argument('--eval_every', type=int, default=100)
    # parser.add_argument('--log_interval', type=int, default=500)
    # parser.add_argument('--audio_window', type=int, default=20480, 
    #                     help='window length to sample from each utterance')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='enables fp16 training')

    parser.add_argument('--load_model', dest='load_model', action='store_true',
                        help="Load the model in model_path to continue a training session")
    args, unparsed = parser.parse_known_args()

    main(args)

