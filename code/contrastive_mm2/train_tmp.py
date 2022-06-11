"""
Contrastive multimodal learning training.
Author: Casper Wortmann
Usage: python train.py --args
"""
import sys
from argparse import ArgumentParser
import time
from time import time
from datetime import datetime
import logging
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
from dataclasses import dataclass
from omegaconf import OmegaConf

import torch
from torch import nn, Tensor
from torch.utils import data as datautil
from torch.utils.data import Sampler, BatchSampler, DataLoader
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

from transformers import AutoTokenizer, AutoModel, AdamW, AutoConfig
from transformers import get_constant_schedule, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

import psutil, gc

import os
# import cv2
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
# import albumentations as A

import torch
from torch import nn
import torch.nn.functional as F
# import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

# Load static configuration variables. 
conf = OmegaConf.load("./config.yaml")
print("[cudacheck] Is cuda available? ", torch.cuda.is_available())

################################################################################
# move to utils.py
def set_seed(args):
    """
    Sets the seed for the current run.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def set_logconfig():
    """
    Setup logging format.
    """
    logging.basicConfig(format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.INFO)

def get_log_name(args, dc):
    """
    Returns name of the current run.
    """
    log_name = "run2-{}_{}_{}_{}_{}_{}".format(args.loss_type, args.text_proj_head, 
            args.audio_proj_head, args.final_projection_dim, dc.pad_pack,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_name = os.path.join(args.log_dir, log_name)
    return log_name

def setup_config(args, dc, device='cpu'):
    """
    Creates a configuration file and saves it to disk. 
    """
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
    pprint(fullcfg)

    # Save config.
    config_dir = os.path.join(log_name, 'config.json')
    with open(config_dir, "w") as file:
        json.dump(config, file, indent=4, sort_keys=True)
    return fullcfg

################################################################################
# Move to dataloader.py
class MMloader(object):
    """
    Module that load datasets. 
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
            train_dataset = self.get_sp_dataset(CFG, directory=conf.sp_sample_path, traintest="train", load_full=self.load_full, device=self.device)
        elif train_dataset_name == "sp":
            train_dataset = self.get_sp_dataset(CFG, directory=conf.sp_path,  traintest="train", load_full=self.load_full, device=self.device)
        else:
            raise Exception('Unknown dataset')
        if CFG.weak_shuffle:
            self.train_loader = DataLoader(
                train_dataset, batch_size=None,  # must be disabled when using samplers
                sampler=BatchSampler(RandomBatchSampler(train_dataset, batch_size), batch_size=batch_size, drop_last=True)
            )
            self.train_dataset = train_dataset
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
            self.train_dataset = train_dataset
            self.train_loader.collate_fn = self.train_dataset.collate_fn
        print("[MMloader] train dataset loaded, length: ", len(self.train_dataset))

        if val_dataset_name == "sp_sample":
            val_dataset = self.get_sp_dataset(CFG, directory=conf.sp_sample_path,  traintest="val", load_full=self.load_full, device=self.device)
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs) 
            # self.val_loader = DataLoader(
            #     val_dataset, batch_size=None,  # must be disabled when using samplers
            #     sampler=BatchSampler(RandomBatchSampler(val_dataset, batch_size), batch_size=batch_size, drop_last=True)
            # )
        elif val_dataset_name == "sp":
            val_dataset = self.get_sp_dataset(CFG, directory=conf.sp_path,  traintest="val",  load_full=self.load_full, device=self.device)
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        else:
            raise Exception('Unknown dataset')
        self.val_dataset = val_dataset
        self.val_loader.collate_fn = self.val_dataset.collate_fn
        print("[MMloader] val dataset loaded, length: ", len(val_dataset))

        if test_dataset_name == "sp_sample":
            test_dataset = self.get_sp_dataset(CFG, directory=conf.sp_sample_path,  traintest="test", load_full=self.load_full, device=self.device)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs) 
        elif test_dataset_name == "sp":
            test_dataset = self.get_sp_dataset(CFG, directory=conf.sp_path,  traintest="test",  load_full=self.load_full, device=self.device)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
        else:
            raise Exception('Unknown dataset')
        self.test_dataset = test_dataset
        self.test_loader.collate_fn = self.test_dataset.collate_fn
        print("[MMloader] test dataset loaded, length: ", len(test_dataset))

    def get_sp_dataset(self, CFG, directory=conf.sp_sample_path, traintest="train", load_full=False, device=None):
        if CFG.weak_shuffle and traintest=='train':
            dataset =  spDatasetWeakShuffle(CFG, directory=directory, traintest=traintest,  load_full=load_full, device=device)
        else:
            dataset =  spDatasetNoMemory(CFG, directory=directory, traintest=traintest,  load_full=load_full, device=device)
        
        return dataset

class spDatasetNoMemory(datautil.Dataset):
    """
    Spotify Podcast dataset dataloader. 
    """
    def __init__(self, CFG, directory=conf.sp_sample_path, traintest="train", load_full=False, device=None):
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
        self.traintest = traintest

        self.tokenizer = AutoTokenizer.from_pretrained(CFG.text_model_name)
        self.text_max_length = CFG.text_max_length

        for h5idx, h5py_file in enumerate(h5py_files):    
            f = h5py.File(h5py_file, 'r')
            self.max_embed_dim = max(self.max_embed_dim, f.attrs['max_embed_dim'])  # TODO: can be removed?
            
            if h5idx % 10 == 0:
                print("[spdataset] loading {}/{}".format(h5idx, len(h5py_files)))

            for sentidx in range(len(f['sentences'])):
                idx2file[sample_idx] = (h5idx, sentidx)
                sample_idx += 1

                if traintest == 'train' and CFG.max_train_samples > 0 and sample_idx >= CFG.max_train_samples:
                    print("[spdataset] Max exceeded: ", sample_idx)
                    f.close()
                    break
            else:
                f.close()
                continue
            break

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
        if self.traintest == 'test':
            h5py_idx, sent_idx = self.idx2file[index]
            h5py_file = self.h5py_idx2file[h5py_idx]

            f = h5py.File(h5py_file, 'r')
            sent = f['sentences'][sent_idx]
            full_embeds = torch.Tensor(np.array(f[str(sent_idx)]))
            target = f['seg_ts'][sent_idx].decode("utf-8") 
            sample = (sent, full_embeds, target)
            f.close()

        elif self.load_full:
            h5py_idx, sent_idx = self.idx2file[index]
            h5py_file = self.h5py_idx2file[h5py_idx]

            f = h5py.File(h5py_file, 'r')

            sent = f['sentences'][sent_idx].decode("utf-8")
            # sent = f['sentences'][sent_idx]
            full_embeds = torch.Tensor(np.array(f[str(sent_idx)]))
            sample = (sent, full_embeds)
            f.close()
            
        else:
            h5py_idx, sent_idx = self.idx2file[index]
            h5py_file = self.h5py_idx2file[h5py_idx]

            f = h5py.File(h5py_file, 'r')
            
            sent = f['sentences'][sent_idx].decode("utf-8")
            # sent = f['sentences'][sent_idx]

            mean_embeds = torch.Tensor(f['mean_embeddings'][sent_idx])
            sample = (sent, mean_embeds)
            f.close()
            
        return sample

    def full_batching_collate(self, batch):
        """ Return a batch containing samples of the SP dataset"""
        text_embeds = []
        audio_embeds = []
        full_text = []
        lengths  = []
        
        for example in batch:
            full_text.append(example[0])
            text_embeds.append(example[0])
            audio_embeds.append(example[1])
            lengths.append(len(example[1]))
        
        # Pad the audio embeddings
        padded_audio_embeds = pad_sequence(audio_embeds, batch_first=True).to(self.device)
        
        # Tokenize text
        text_embeds = self.tokenizer(
            text_embeds, padding=True, truncation=True, max_length=self.text_max_length, return_tensors='pt'
        ).to(self.device)

        if self.traintest == 'test':
            targs = []
            for example in batch:
                targs.append(example[2])
            return text_embeds, padded_audio_embeds, lengths, targs, full_text
        else:
            return text_embeds, padded_audio_embeds, lengths

    def mean_batching_collate(self, batch):
        """ 
        Returns a batch containing samples of the SP dataset. Audio embeddings
        are averaged to fit into a non-sequential module. 
        """
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


class spDatasetWeakShuffle(datautil.Dataset):
    """
    Spotify Podcast dataset dataloader. 
    Weak shuffling to enable shuffle while clustering sentences of similar length.
    """
    def __init__(self, CFG, directory=conf.sp_sample_path, traintest="train", load_full=False, device=None):
        print("[spDataset] init from directory [WEAKSHUFFLE] ", directory, traintest)
        directory = os.path.join(directory, traintest)
        h5py_files = list(Path(directory).glob('*.h5'))
        print("[spDataset] found {} h5py files".format(len(h5py_files)))
        self.max_embed_dim = 0
        self.device = device

        idx2file = {}
        self.h5py_idx2file = h5py_files
        sample_idx = 0

        self.load_full = load_full
        self.traintest = traintest

        self.tokenizer = AutoTokenizer.from_pretrained(CFG.text_model_name)
        self.text_max_length = CFG.text_max_length

        for h5idx, h5py_file in enumerate(h5py_files):    
            f = h5py.File(h5py_file, 'r')
            print(h5py_file)
            self.max_embed_dim = max(self.max_embed_dim, f.attrs['max_embed_dim'])  # TODO: can be removed?
            
            if h5idx % 10 == 0:
                print("[spdataset] loading {}/{}".format(h5idx, len(h5py_files)))

            for sentidx in range(len(f['sentences'])):
                idx2file[sample_idx] = (h5idx, sentidx)
                sample_idx += 1

                if CFG.max_train_samples > 0 and traintest == 'train' and sample_idx >= CFG.max_train_samples:
                    print("[del] Max exceeded {}".format(sample_idx))
                    f.close()
                    break
            else:
                f.close()
                continue
            break

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
        if self.traintest == 'test':
            print("Weak shuffling not supported for test set")
            raise NotImplementedError

        elif self.load_full:
            text_embeds = []
            audio_embeds = []
            full_text = []
            lengths  = []

            last_idx = -1
            del_lens = []
            del_lens2 = []

            for enum, i in enumerate(index):
                h5py_idx, sent_idx = self.idx2file[i]

                if h5py_idx != last_idx:
                    if enum != 0:
                        f.close()
                    h5py_file = self.h5py_idx2file[h5py_idx]
                    f = h5py.File(h5py_file, 'r')

                # sent = f['sentences'][sent_idx]
                sent = f['sentences'][sent_idx].decode("utf-8")
                full_embeds = torch.Tensor(np.array(f[str(sent_idx)]))

                del_lens.append(len(sent))
                del_lens2.append(full_embeds.shape[0])

                # Collate fn
                full_text.append(sent)
                text_embeds.append(sent)
                audio_embeds.append(full_embeds)
                lengths.append(len(full_embeds))

            f.close()
             # Pad the audio embeddings
            padded_audio_embeds = pad_sequence(audio_embeds, batch_first=True).to(self.device)
            
            # Tokenize text
            text_embeds = self.tokenizer(
                text_embeds, padding=True, truncation=True, max_length=self.text_max_length, return_tensors='pt'
            ).to(self.device)

            return text_embeds, padded_audio_embeds, lengths

        else:
            text_embeds = []
            audio_embeds = []
            lengths  = []
            last_idx = -1

            for enum, i in enumerate(index):
                h5py_idx, sent_idx = self.idx2file[i]

                if h5py_idx != last_idx:
                    if enum != 0:
                        f.close()
                    h5py_file = self.h5py_idx2file[h5py_idx]
                    f = h5py.File(h5py_file, 'r')

                sent = f['sentences'][sent_idx].decode("utf-8")
                mean_embeds = torch.Tensor(f['mean_embeddings'][sent_idx])

                text_embeds.append(sent)
                audio_embeds.append(mean_embeds)

            f.close()

            # Combine audio embeddings to a Tensor
            audio_embeds = torch.stack(audio_embeds).to(self.device)

            # Tokenize text
            max_length = 32 # TODO: is dit nodig?
            text_embeds = self.tokenizer(
                text_embeds, padding=True, truncation=True, max_length=max_length, return_tensors='pt'
            ).to(self.device)

            return text_embeds, audio_embeds, lengths
     


class RandomBatchSampler(Sampler):
    """Sampling class to create random sequential batches from a given dataset
    E.g. if data is [1,2,3,4] with bs=2. Then first batch, [[1,2], [3,4]] then shuffle batches -> [[3,4],[1,2]]
    This is useful for cases when you are interested in 'weak shuffling'
    :param dataset: dataset you want to batch
    :type dataset: torch.utils.data.Dataset
    :param batch_size: batch size
    :type batch_size: int
    :returns: generator object of shuffled batch indices
    """
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.batch_ids = torch.randperm(int(self.n_batches))

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for id in self.batch_ids:
            idx = torch.arange(id * self.batch_size, (id + 1) * self.batch_size)
            t = torch.randperm(idx.shape[0])
            idx= idx[t].view(idx.size())
            for index in idx:
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(int(self.n_batches) * self.batch_size, self.dataset_length)
            for index in idx:
                yield int(index)


# ################################################################################
# # Textmodules
# class TextEncoder(nn.Module):
#     def __init__(self, CFG):
#         super().__init__()
#         model_name_or_path = CFG.text_model_name
#         self.config_keys = ['max_seq_length', 'do_lower_case']
#         self.do_lower_case = None

#         config = AutoConfig.from_pretrained(model_name_or_path)
#         self.hidden_dim = config.dim
#         print("[Transformer_multimodal] Configurations used: \n", config)
        
#         # This is self._load_model in mm2
#         self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config)
#         self.max_seq_length = CFG.text_max_length

#         # we are using the CLS token hidden representation as the sentence's embedding
#         self.target_token_idx = 0
#         self.pooling = CFG.text_pooling # TODO: delete this line?

#     def forward(self, text_embeds):
#         trans_features = {'input_ids': text_embeds['input_ids'], 'attention_mask': text_embeds['attention_mask']}
#         output = self.auto_model(**trans_features, return_dict=False)

#         output_tokens = output[0]
#         features = {'input_ids':text_embeds['input_ids'], 'attention_mask':text_embeds['attention_mask'],'token_embeddings': output_tokens}
#         return features

#     def get_word_embedding_dimension(self) -> int:
#         return self.auto_model.config.hidden_size


# class text_ProjectionHead(nn.Module):
#     # TODO: This module is depricated.
#     def __init__(self, CFG):

#         super().__init__()
#         embedding_dim=CFG.mutual_embedding_dim
#         projection_dim=CFG.final_projection_dim
#         dropout=CFG.text_dropout

#         self.net = nn.Sequential(
#             nn.Linear(embedding_dim, projection_dim),
#             nn.GELU(),
#             nn.Linear(projection_dim, projection_dim),
#             nn.Dropout(dropout),
#             nn.LayerNorm(projection_dim),
#         )
    
#     def forward(self, x):
#         output_tokens = self.net(x['token_embeddings'])
#         features = {'input_ids':x['input_ids'], 'attention_mask':x['attention_mask'],'token_embeddings': output_tokens}
#         return features

# class Pooling(nn.Module):
#     """Performs pooling (max or mean) on the token embeddings.

#     Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
#     You can concatenate multiple poolings together.

#     :param word_embedding_dimension: Dimensions for the word embeddings
#     :param pooling_mode: Can be a string: mean/max/cls. If set, overwrites the other pooling_mode_* settings
#     :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
#     :param pooling_mode_max_tokens: Use max in each dimension over all tokens.
#     :param pooling_mode_mean_tokens: Perform mean-pooling
#     :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but devide by sqrt(input_length).
#     """
#     def __init__(self,
#                  word_embedding_dimension: int = 768,
#                  pooling_mode: str = None,
#                  pooling_mode_cls_token: bool = False,
#                  pooling_mode_max_tokens: bool = False,
#                  pooling_mode_mean_tokens: bool = True,
#                  pooling_mode_mean_sqrt_len_tokens: bool = False,
#                  ):
#         super(Pooling, self).__init__()

#         self.config_keys = ['word_embedding_dimension',  'pooling_mode_cls_token', 'pooling_mode_mean_tokens', 'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens']

#         if pooling_mode is not None:        #Set pooling mode by string
#             pooling_mode = pooling_mode.lower()
#             assert pooling_mode in ['mean', 'max', 'cls']
#             pooling_mode_cls_token = (pooling_mode == 'cls')
#             pooling_mode_max_tokens = (pooling_mode == 'max')
#             pooling_mode_mean_tokens = (pooling_mode == 'mean')

#         self.word_embedding_dimension = word_embedding_dimension
#         self.pooling_mode_cls_token = pooling_mode_cls_token
#         self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
#         self.pooling_mode_max_tokens = pooling_mode_max_tokens
#         self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens

#         pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens, pooling_mode_mean_sqrt_len_tokens])
#         self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

#     def __repr__(self):
#         return "Pooling({})".format(self.get_config_dict())

#     def get_pooling_mode_str(self) -> str:
#         """
#         Returns the pooling mode as string
#         """
#         modes = []
#         if self.pooling_mode_cls_token:
#             modes.append('cls')
#         if self.pooling_mode_mean_tokens:
#             modes.append('mean')
#         if self.pooling_mode_max_tokens:
#             modes.append('max')
#         if self.pooling_mode_mean_sqrt_len_tokens:
#             modes.append('mean_sqrt_len_tokens')

#         return "+".join(modes)

#     def forward(self, features: Dict[str, Tensor]):
#         token_embeddings = features['token_embeddings']
#         attention_mask = features['attention_mask']

#         ## Pooling strategy
#         output_vectors = []
#         if self.pooling_mode_cls_token:
#             cls_token = features.get('cls_token_embeddings', token_embeddings[:, 0])  # Take first token by default
#             output_vectors.append(cls_token)
#         if self.pooling_mode_max_tokens:
#             input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#             token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
#             max_over_time = torch.max(token_embeddings, 1)[0]
#             output_vectors.append(max_over_time)
#         if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
#             input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#             sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

#             #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
#             if 'token_weights_sum' in features:
#                 sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
#             else:
#                 sum_mask = input_mask_expanded.sum(1)

#             sum_mask = torch.clamp(sum_mask, min=1e-9)

#             if self.pooling_mode_mean_tokens:
#                 output_vectors.append(sum_embeddings / sum_mask)
#             if self.pooling_mode_mean_sqrt_len_tokens:
#                 output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

#         output_vector = torch.cat(output_vectors, 1)
#         features.update({'sentence_embedding': output_vector})
#         return features

#     def get_sentence_embedding_dimension(self):
#         return self.pooling_output_dimension

#     def get_config_dict(self):
#         return {key: self.__dict__[key] for key in self.config_keys}

#     def save(self, output_path):
#         with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
#             json.dump(self.get_config_dict(), fOut, indent=2)

#     @staticmethod
#     def load(input_path):
#         with open(os.path.join(input_path, 'config.json')) as fIn:
#             config = json.load(fIn)
#         return Pooling(**config)

# ################################################################################
# # Audiomodules
# class SequentialAudioModel(nn.Module):
#     def __init__(self, CFG):
#         super(SequentialAudioModel, self).__init__()
#         # Defining the number of layers and the nodes in each layer
#         self.hidden_dim = CFG.audio_hidden_dim
#         self.layer_dim = CFG.audio_layer_dim
#         self.device = CFG.device
#         self.audio_model = CFG.audio_proj_head

#         # RNN layers
#         if self.audio_model == 'rnn':
#             self.seq_model = nn.RNN(CFG.audio_encoder_input, 
#                     CFG.audio_hidden_dim, CFG.audio_layer_dim, 
#                     batch_first=True, dropout=CFG.audio_dropout)
#         elif self.audio_model == 'gru':
#             self.seq_model = nn.GRU(
#                     input_size=CFG.audio_encoder_input, 
#                     hidden_size=CFG.audio_hidden_dim, num_layers=CFG.audio_layer_dim, 
#                     batch_first=True, dropout=CFG.audio_dropout)

#         # Fully connected layer
#         self.fc = nn.Linear(CFG.audio_hidden_dim, CFG.mutual_embedding_dim)
#         self.softmax = nn.Softmax(dim=1)

#         # pad_pack = args.pad_pack
#         self.pad_pack = CFG.pad_pack

#     def forward(self, audio_seq):
#         features, length = audio_seq

#         # Initializing hidden state for first input with zeros
#         h0 = torch.zeros(self.layer_dim, features.size(0), self.hidden_dim).requires_grad_().to(self.device)

#         if length != None:
#             if self.pad_pack:
#                 # Pack the features such that we do not compute zero products
#                 features = pack_padded_sequence(features, length, batch_first=True, enforce_sorted=False)

#             out, h0 = self.seq_model(features, h0.detach())
#             if self.pad_pack:
#                 out, output_lengths = pad_packed_sequence(out, batch_first=True)
#             do_trick = True

#         else:
#             # Forward propagation by passing in the input and hidden state into the model
#             out, h0 = self.seq_model(features, h0.detach())     # shape [bs, sl, hidden]
#             do_trick=False

#         # Convert the final state to our desired output shape (batch_size, output_dim)
#         if do_trick:
#             out = h0[-1, :, :]
#         else:
#             # Reshaping the outputs
#             # so that it can fit into the fully connected layer
#             out = out[:, -1, :]     # shape [bs * hidden]

#         out = self.fc(out)
#         out = self.softmax(out)
#         return out  # shape [bs*output]

# class simple_ProjectionHead(nn.Module):
#     """
#     :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
#     """
#     def __init__(self, CFG):
#         super(simple_ProjectionHead, self).__init__()
#         self.input_dim = CFG.audio_encoder_input
#         self.hidden_dim = CFG.audio_hidden_dim
#         self.output_dim = CFG.final_projection_dim
#         self.activation  = CFG.audio_activation
#         self.config_keys = ['input_dim', 'hidden_dim', 'output_dim', 'activation']
#         dropout = CFG.text_dropout

#         if self.activation == 'relu':
#             self.simple_model = nn.Sequential(
#                 nn.Linear(self.input_dim, self.hidden_dim),
#                 nn.BatchNorm1d(num_features=self.hidden_dim), # TODO: experiment with batchnormalization
#                 nn.ReLU(),
#                 nn.Linear(self.hidden_dim, self.output_dim),
#             )
#         else:
#             self.simple_model = nn.Sequential(
#                 nn.Linear(self.input_dim, self.hidden_dim),
#                 nn.GELU(),
#                 nn.Linear(self.hidden_dim, self.output_dim),
#                 nn.Dropout(dropout),
#                 nn.LayerNorm(self.output_dim),
#             )

#     def __repr__(self):
#         return "Simple_projection_head({}) ".format(self.get_config_dict())

#     def get_config_dict(self):
#         return {key: self.__dict__[key] for key in self.config_keys}

#     def forward(self, feat_len):
#         """Returns token_embeddings, cls_token"""

#         features, _ = feat_len
#         features = self.simple_model(features)

#         return features

# class mmModule(nn.Module):
#     def __init__(self, CFG):
#         super().__init__()
#         self.batch_size = CFG.batch_size
#         self.device = CFG.device

#         self.loss_type = CFG.loss_type

#         if CFG.text_proj_head == 'sph':
#             text_encoder = TextEncoder(CFG)
#             text_projection = text_ProjectionHead(CFG)
#             if CFG.text_pooling == 'mean':
#                 pooling_model = Pooling(text_encoder.get_word_embedding_dimension(),
#                 pooling_mode_mean_tokens = True)
#             elif CFG.text_pooling == 'cls':
#                 pooling_model = Pooling(text_encoder.get_word_embedding_dimension(),
#                 pooling_mode_mean_tokens = False,
#                 pooling_mode_cls_token = True)

#             text_modules = [text_encoder, text_projection, pooling_model]

#         elif CFG.text_proj_head.lower() == 'none':
#             text_encoder = TextEncoder(CFG)
#             if CFG.text_pooling == 'mean':
#                 pooling_model = Pooling(text_encoder.get_word_embedding_dimension(),
#                 pooling_mode_mean_tokens = True)
#             elif CFG.text_pooling == 'cls':
#                 pooling_model = Pooling(text_encoder.get_word_embedding_dimension(),
#                 pooling_mode_mean_tokens = False,
#                 pooling_mode_cls_token = True)

#             pooling_model = Pooling(text_encoder.get_word_embedding_dimension())
#             text_modules = [text_encoder, pooling_model]

#         if CFG.audio_proj_head in ['rnn', 'gru']:
#             audio_encoder = SequentialAudioModel(CFG)
#             audio_modules = [audio_encoder]
#         elif CFG.audio_proj_head in ['sph']:
#             audio_encoder = simple_ProjectionHead(CFG)
#             audio_modules = [audio_encoder]
        
#         # Create the full model
#         if text_modules is not None and not isinstance(text_modules, OrderedDict):
#             text_modules = OrderedDict([(str(idx), module) for idx, module in enumerate(text_modules)])
#         if audio_modules is not None and not isinstance(audio_modules, OrderedDict):
#             audio_modules = OrderedDict([(str(idx), module) for idx, module in enumerate(audio_modules)])

#         self.text_model = nn.Sequential(text_modules)
#         self.audio_model = nn.Sequential(audio_modules)

#         self.eval_every = CFG.eval_every        # TODO: depr?
#         self.print_every = CFG.print_every
#         self.batch_size = CFG.batch_size
#         self.device = CFG.device

#         self.log_name = CFG.log_name
#         self.init_logging()

#         self.best_loss = float('inf')
#         self.max_grad_norm = 1 # magic number for now

#     @staticmethod
#     def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
#         """
#         Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
#         """
#         scheduler = scheduler.lower()
#         if scheduler == 'constantlr':
#             return get_constant_schedule(optimizer)
#         elif scheduler == 'warmupconstant':
#             return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
#         elif scheduler == 'warmuplinear':
#             return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
#         else:
#             raise ValueError("Unknown scheduler {}".format(scheduler))

#     def _get_optimizer(self, loss_model):
#         # Prepare optimizers
#         param_optimizer = list(loss_model.named_parameters())
#         no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#         optimizer_grouped_parameters = [
#             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
#             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#         ]
#         optimizer_params = self.optimizer_params
#         optimizer = self.optimizer_class(optimizer_grouped_parameters, **optimizer_params)

#         return optimizer

#     def fit(self,
#         CFG,
#         train_loader,
#         val_loader,
#         steps_per_epoch,
#         loss_model,
#         start_epoch=1,
#         optimizer_class: Type[Optimizer] = AdamW,
#         fstep = 0,
#         loaded_optimizer_state = None,
#         loaded_sched_state = None,
#         ):
#         self.text_model.to(self.device)
#         self.audio_model.to(self.device)
#         loss_model.to(self.device)

#         self.val_loader = val_loader
#         num_train_steps = int(steps_per_epoch * CFG.num_epochs)
#         warmup_steps =  math.ceil(steps_per_epoch *  CFG.num_epochs * 0.1)  
#         self.weight_decay = CFG.weight_decay
#         self.optimizer_class = optimizer_class
#         self.optimizer_params = {'lr': CFG.lr}
#         scheduler_method='WarmupLinear' 

#         steps_so_far = (start_epoch + 1) * fstep
#         self.num_train_steps = num_train_steps

#         # Initiate or load an optimizer
#         if loaded_optimizer_state == None:
#             optimizer = self._get_optimizer(loss_model)
#         else:
#             optimizer = self._get_optimizer(loss_model)
#             optimizer.load_state_dict(loaded_optimizer_state)

#         # Initiate or load a scheduler
#         if loaded_sched_state == None:
#             scheduler = self._get_scheduler(optimizer, scheduler=scheduler_method, warmup_steps=warmup_steps, t_total=num_train_steps)
#         else:
#             scheduler = self._get_scheduler(optimizer, scheduler=scheduler_method, warmup_steps=warmup_steps, t_total=num_train_steps)
#             scheduler.load_state_dict(loaded_sched_state)

#         print("[del] eval_every, print_every, warmup_steps: ", self.eval_every, self.print_every, warmup_steps)

#         time_del = []
#         for epoch in range(start_epoch, CFG.num_epochs):
#             t1 = time()
#             loss_model.zero_grad()
#             loss_model.train()
#             # test1 = time()

#             # Full training
#             # TODO: if training from checkpoint, stop batch in time
#             for step, batch in enumerate(iter(train_loader)):
                
#                 # ### Leave for debugging, check speedup weak shuffling.
#                 # time_del.append(time() - test1)
#                 # # sent_features, audio_features, seq_len = batch
#                 # if step % 10 == 0 and step != 0:
#                 #     print("avg:", np.mean(time_del))
#                 #     time_del = []
#                 #     test1 = time()
#                 # else:
#                 #     # print("continue ", step)
#                 #     test1 = time()
#                 # continue
                
#                 if step < fstep:
#                     print("[DEBUG] loading checkpoint, continue")
#                     continue

#                 if  step % self.eval_every == 0 or step == steps_per_epoch - 1: 
#                     # Evaluate on validation set. 
#                     print("[eval] start evaluation")
#                     mean_loss, metrics = self.evaluate(loss_model)
#                     self.add_logging(epoch, steps_so_far, mean_loss, metrics, train=False)
                    
#                     print("[eval] Epoch {} Step {}/{} \t loss {} \t acc {}".format(epoch, step, num_train_steps, mean_loss, metrics['mean_acc']))
#                     if mean_loss < self.best_loss:
#                         print("[eval] better model found")
#                         self.best_loss = mean_loss 
#                         if args.save_model:
#                             self.save_model()
#                         if args.save_checkpoint:
#                             self.save_checkpoint(epoch, step, optimizer, scheduler)

#                     # If csv files are properly set up, save plots.
#                     if os.path.isfile(self.train_csv_filename):
#                         self.output_all_plots()

#                 sent_features, audio_features, seq_len = batch
#                 loss_value, metrics = loss_model(sent_features, audio_features, seq_len)

#                 loss_value.backward()
#                 torch.nn.utils.clip_grad_norm_(loss_model.parameters(), self.max_grad_norm)
#                 optimizer.step()
#                 optimizer.zero_grad()

#                 scheduler.step()
#                 steps_so_far += 1

#                 if step % self.print_every == 0:
#                     print("[fit] Epoch {} Step {}/{} \t loss {} \t acc {}".format(epoch, step, num_train_steps, loss_value.item(), metrics['mean_acc']))
#                     self.add_logging(epoch, steps_so_far, loss_value.item(), metrics, train=True)

#             self.output_all_plots()
#             t2 = time()
#             print("[fit] epoch duration {} seconds".format(int(t2-t1)))
#             if args.save_model:
#                 self.save_model("_epoch_"+str(epoch))

#         print("[fit] Done training")

#     def evaluate(self, loss_model):
#         loss_model.eval()
#         losses = 0
#         self.to(self.device)

#         iterator = iter(self.val_loader)
#         total_len = 500   # for now I evaluate on a subset

#         with torch.no_grad():
#             for step in range(total_len):
#                 batch = next(iterator)
#                 sent_features, audio_features, seq_len  = batch
#                 with torch.no_grad():
#                     loss_value, metrics = loss_model(sent_features, audio_features, seq_len)
#                     losses += loss_value.detach().cpu().item()

#                     if step == 0:
#                         met_sum = Counter(metrics.copy())
#                     else:
#                         met_sum.update(Counter(metrics))

#         mean_metrics = {k: value / total_len  for k, value in met_sum.items()}
#         mean_loss = losses / total_len

#         del met_sum
#         loss_model.train()
#         return mean_loss, mean_metrics

#     def output_all_plots(self):
#         to_plot(self.train_csv_filename, column='audio_acc', title="Train accuracy (audio)")
#         to_plot(self.train_csv_filename, column='text_acc', title="Train accuracy (text)")
#         to_plot(self.train_csv_filename, column='loss', title="Train loss")
#         to_plot(self.eval_csv_filename, column='mean_acc', title="val accuracy (mean)")
  
#     def init_logging(self):
#         self.model_save_path = '{}/output'.format(self.log_name)
#         os.makedirs(self.model_save_path, exist_ok=True)
#         self.train_csv_filename = os.path.join(self.model_save_path, "train.csv")
#         self.eval_csv_filename = os.path.join(self.model_save_path, "val.csv")

#     def init_csv(self, metrics):
#         for filename in [self.train_csv_filename, self.eval_csv_filename]:
#             self.metric_headers = [metric for metric in metrics.keys()]
#             self.train_csv_headers = ["epoch", "steps", "loss"] + self.metric_headers
#             with open(filename, newline='', mode='w', encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(self.train_csv_headers)

#     def add_logging(self, epoch, steps, loss, metrics, train=True):
#         if train:
#             filename = self.train_csv_filename
#         else:
#             filename = self.eval_csv_filename

#         output_file_exists = os.path.isfile(filename)
#         if not output_file_exists:
#             self.init_csv(metrics)
            
#         # Add metrics to CSV
#         metric_vals = [metrics[header] for header in self.metric_headers]
#         with open(filename, newline='', mode="a", encoding="utf-8") as f:
#             writer = csv.writer(f)
#             writer.writerow([epoch, steps, loss] + metric_vals)

#     def save_model(self, extra_name=""):
#         # Save the model
#         output_dir = os.path.join(self.model_save_path, '{}{}_weights.pth'.format('full_model', extra_name))
#         torch.save(self.state_dict(), output_dir)

#     def save_checkpoint(self, epoch, step, optimizer, scheduler):
#         checkpoint = { 
#             'epoch': epoch,
#             'step': step,
#             'full_model': self,
#             'optimizer': optimizer,
#             'lr_sched': scheduler.state_dict()
#         }
#         output_dir = os.path.join(self.model_save_path, 'checkpoint.pth')
#         torch.save(checkpoint, output_dir)


# def dek_cos_sim(a: Tensor, b: Tensor):
#     """
#     Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
#     :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
#     """
#     if not isinstance(a, torch.Tensor):
#         a = torch.tensor(a)

#     if not isinstance(b, torch.Tensor):
#         b = torch.tensor(b)

#     if len(a.shape) == 1:
#         a = a.unsqueeze(0)

#     if len(b.shape) == 1:
#         b = b.unsqueeze(0)

#     a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
#     b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
#     return torch.mm(a_norm, b_norm.transpose(0, 1))


# class multimodal_loss(nn.Module):
#     """
#         This loss expects as input a batch consisting of ... etc
#     """
#     def __init__(self, full_model, CFG):
#         """
#         test
#         """
#         super(multimodal_loss, self).__init__()

#         self.text_model = full_model.text_model
#         self.audio_model = full_model.audio_model
#         self.loss_type = CFG.loss_type
#         self.scale_type = CFG.scale_type
#         self.scale = CFG.scale

#         if self.scale_type == 'fixed':
#             self.fixed_scale = self.scale
#         elif self.scale_type == 'learned':
#             self.logit_scale = nn.Parameter(torch.log(torch.ones([]) * 100))
#             self.logit_scale.requires_grad = True

#         self.similarity_fct = dek_cos_sim
#         self.cross_entropy_loss = nn.CrossEntropyLoss()

#         self.device = torch.device(CFG.device)
#         self.batch_size = full_model.batch_size

#     def init_parameters_logtiscale(self):
#         nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

#     def forward(self, sentence_features: Iterable[Dict[str, Tensor]], audio_features: Tensor, seq_len):
#         # Get sentence Representations (shape [batchsize, 768])
#         reps_text = self.text_model(sentence_features)['sentence_embedding']

#         # Get Audio representations
#         reps_audio = self.audio_model((audio_features, seq_len))

#         if self.loss_type == 'clip_loss_old':
#             # Loss function from CLIP paper
#             if self.scale_type == 'fixed':
#                 audio_logits =  (reps_audio @ reps_text.t()) * self.fixed_scale
#             elif self.scale_type == 'learned':
#                 cur_logit_scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
#                 audio_logits =  (reps_audio @ reps_text.t()) * cur_logit_scale.exp()
#             text_logits = audio_logits.t()

#             audio_logits = audio_logits / audio_logits.norm(dim=1, keepdim=True)
#             text_logits = text_logits / text_logits.norm(dim=1, keepdim=True)

#             ground_truth = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
#             total_loss = F.cross_entropy(audio_logits,ground_truth, weight=None) + F.cross_entropy(text_logits.transpose(-1, -2), ground_truth, weight=None)
#             metrics = get_metrics(audio_logits.detach(), text_logits.detach(), ground_truth)
   
#             return total_loss, metrics

#         if self.loss_type == 'clip_loss':
#             # Normalise features
#             reps_audio = reps_audio / reps_audio.norm(dim=1, keepdim=True)
#             reps_text = reps_text / reps_text.norm(dim=1, keepdim=True)

#             # Loss function from CLIP paper
#             if self.scale_type == 'fixed':
#                 audio_logits =  (reps_audio @ reps_text.t()) * self.fixed_scale
#             elif self.scale_type == 'learned':
#                 cur_logit_scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
#                 audio_logits =  (reps_audio @ reps_text.t()) * cur_logit_scale.exp()

#             text_logits = audio_logits.t()

#             # Calculate metrics
#             ground_truth = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
#             total_loss = F.cross_entropy(audio_logits,ground_truth, weight=None) + F.cross_entropy(text_logits.transpose(-1, -2), ground_truth, weight=None)
#             metrics = get_metrics(audio_logits.detach(), text_logits.detach(), ground_truth)
   
#             return total_loss, metrics
    
#         elif self.loss_type == 'simcse_loss':
#             # SIMCSE-based NCE loss
#             if self.scale_type == 'fixed':
#                 scores = self.similarity_fct(reps_text, reps_audio) * self.fixed_scale
#             elif self.scale_type == 'learned':
#                 cur_logit_scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
#                 scores = self.similarity_fct(reps_text, reps_audio) * cur_logit_scale.exp()

#             labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
#             loss = self.cross_entropy_loss(scores, labels)

#             metrics = get_metrics(scores.detach(), scores.t().detach(), labels)
#             return loss, metrics

#     def get_config_dict(self):
#         return {
#             'scale_type': self.scale_type, 
#             'similarity_fct': self.similarity_fct.__name__
#         }

# def get_metrics(audio_logits, text_logits, ground_truth):
#     # TODO: naar self.multimodal_loss
#     metrics ={}
#     logits = {"audio": audio_logits, "text": text_logits}

#     accs = []
#     for name, logit in logits.items():
#         acc = torch.mean((logit.argmax(dim=-1) == ground_truth).float()).item()
#         accs.append(acc)
#         metrics[f"{name}_acc"] = acc
#         ranking = torch.argsort(logit, descending=True)
#         preds = torch.where(ranking == ground_truth)[1]
#         preds = preds.detach().cpu().numpy()
#         if len(preds) > 0:
#             metrics[f"{name}_mean_rank"] = preds.mean() + 1
#             for k in [1, 5, 10]:
#                 metrics[f"{name}_R@{k}"] = np.mean(preds < k)
#         else:
#             metrics[f"{name}_mean_rank"] = 0.0
#             for k in [1, 5, 10]:
#                 metrics[f"{name}_R@{k}"] = 0.0
#     metrics['mean_acc'] = np.mean(accs)
#     return metrics

# def cross_entropy(preds, targets, reduction='none'):
#     log_softmax = nn.LogSoftmax(dim=-1)
#     loss = (-targets * log_softmax(preds)).sum(1)
#     if reduction == "none":
#         return loss
#     elif reduction == "mean":
#         return loss.mean()

# def to_plot(filename, column='accuracy', title="Val accuracy"):
#     csv_file = pd.read_csv(filename)

#     ax = sns.lineplot(x=csv_file.steps, y=csv_file[column])
#     ax.set(title=title)
    
#     output_dir = os.path.split(filename)[0]
#     output_title = os.path.split(filename)[-1].split('.')[0]
#     output_name = os.path.join(output_dir, output_title + "_" + column + '.jpg')
#     plt.savefig(output_name)
#     plt.close()

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
    eval_every: int = 1
    print_every: int = 1
    log_name: str = ''

    train_dataset: str = ''
    val_dataset: str = ''
    test_dataset: str = ''
    seed: int = 100

    max_train_samples: int = 0
    save_model: bool = False
    save_checkpoint: bool = False
    load_model_path: str = ''
    load_model: bool = False
    load_checkpoint: bool = False
    load_checkpoint_path: str = ''
    weak_shuffle: bool = False

    # NEW
    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0
    batch_size = 32
    num_workers = 4
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 768
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

class AudioEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """
    def __init__(
        self, model_name="ImageEncoder", pretrained=Cfg.pretrained, trainable=Cfg.trainable
    ):
        super().__init__()

        print("Would create model for name: ", model_name)
        # self.model = timm.create_model(
        #     model_name, pretrained, num_classes=0, global_pool="avg"
        # )
        self.input_dim = 1024
        self.hidden_dim = 768
        self.output_dim = 768
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(num_features=self.hidden_dim), # TODO: experiment with batchnormalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        #x, _ = feat_len
        return self.model(x)

class AudioEncoderRNN(nn.Module):
    def __init__(self):
        super(AudioEncoderRNN, self).__init__()

        self.input_dim = 1024
        self.hidden_dim = 768
        self.dropout = 0.1


        #self.embedding = nn.Embedding(self.input_dim , self.hidden_dim)
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(self.dropout)

        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.batch_size = 128

    def forward(self, input):

        hidden = self.initHidden()
        print("Input forward: ", input.shape)

        # for i in range(input.shape[1]):
        #     print("i=", i)
        #     cur_embed = input[:,i,:]
        #     print("Cur embed: ", cur_embed.shape)
        #     #cur_embed = cur_embed.float()
        #     # print("Cur embed: ", cur_embed)
        #     #embedded = self.embedding(cur_embed)
        #     # print("Embedded: ", embedded.shape)
        #     #output = embedded.view(1, 1, -1)
            # print("OUtput: ", output.shape)
        output, hidden = self.gru(input, hidden)

        # TODO: this is the same result, but a different backward propagation.
        # Option 1:
        output = self.fc(self.relu(output[:,-1]))
        # Option 2:
        # output = self.fc(self.relu(output[:, -1, :]))
        return output

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_dim, device=Cfg.device)


class TextEncoder(nn.Module):
    def __init__(self, model_name=Cfg.text_encoder_model, pretrained=Cfg.pretrained, trainable=Cfg.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=Cfg.projection_dim,
        dropout=Cfg.dropout
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
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
    def __init__(
        self, 
        temperature=Cfg.temperature,
        image_embedding=Cfg.image_embedding,
        text_embedding=Cfg.text_embedding,
    ):
        super().__init__()
        self.audio_proj_head = args.audio_proj_head
        if self.audio_proj_head == 'sph':
            self.audio_encoder = AudioEncoder()
        elif self.audio_proj_head == 'gru':
            self.audio_encoder = AudioEncoderRNN()
        else: 
            raise NotImplementedError
        self.text_encoder = TextEncoder()
        self.audio_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature


        

    def forward(self, batch):
        sent_features, audio_features, seq_len = batch

        # Getting Image and Text Features
        if self.audio_proj_head == 'sph':
            audio_features = self.audio_encoder(audio_features)
        elif self.audio_proj_head == 'gru':
            audio_features = self.audio_encoder(audio_features)
            
        text_features = self.text_encoder(
            input_ids=sent_features["input_ids"], attention_mask=sent_features["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        audio_embeddings = self.audio_projection(audio_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ audio_embeddings.T) / self.temperature
        audio_similarity = audio_embeddings @ audio_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (audio_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        ##batch = {k: v.to(Cfg.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = Cfg.batch_size
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    count = 0
    for batch in tqdm_object:
        # batch = {k: v.to(Cfg.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = Cfg.batch_size
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

        count += 1
        if count > 500:
            print("[del] break for now")
            break
    return loss_meter


def main(args):
    # Setup configuration.
    t_start = time()
    set_logconfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FullCfg = setup_config(args, Cfg, device)

    tokenizer = DistilBertTokenizer.from_pretrained(FullCfg.text_tokenizer)


    # Setup dataloaders.
    data_loader = MMloader(FullCfg)
    train_loader = data_loader.train_loader
    val_loader = data_loader.val_loader
    test_loader = data_loader.test_loader # Not used!
    steps_per_epoch = int(len(data_loader.train_dataset) / FullCfg.batch_size)

    model = CLIPModel().to(FullCfg.device)

    params = [
        {"params": model.audio_encoder.parameters(), "lr": FullCfg.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": FullCfg.text_encoder_lr},
        {"params": itertools.chain(
            model.audio_projection.parameters(), model.text_projection.parameters()
        ), "lr": FullCfg.head_lr, "weight_decay": FullCfg.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=FullCfg.patience, factor=FullCfg.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(FullCfg.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        print("Back from train_epoch loss: ", train_loss)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, val_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)

    print("Done for now")
    exit(1)
    best_results(full_model.eval_csv_filename, dur, FullCfg.log_name)


def best_results(eval_csv_filename, dur, out_dir):
    csv_file = pd.read_csv(eval_csv_filename)
    best_idx = csv_file['mean_acc'].idxmax()
    best2 = {'duration': dur}
    for k, v in csv_file.iloc[best_idx].to_dict().items():
        best2[k] = v

    outfile = os.path.join(out_dir, 'best_results.json')
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
    parser.add_argument('--loss_type', default='simcse_loss', const='simcse_loss',
                    nargs='?', choices=['simcse_loss', 'clip_loss', 'clip_loss_simple'],
                    help='Name of scale_type (default: %(default)s)')
    parser.add_argument('--audio_proj_head', default='gru', const='gru',
                    nargs='?', choices=['sph', 'rnn', 'gru'],
                    help='Activation to use in simple proj head (default: %(default)s)')
    parser.add_argument('--text_proj_head', default='None', const='None',
                    nargs='?', choices=['sph', 'None'],
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

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
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

    parser.add_argument('--pad_pack', dest='pad_pack', action='store_true',     # TODO: depricated
                        help="test: use old or new opt.")
    parser.add_argument('--max_train_samples', type=int, default=0,
                        help='Fixed scale to use')
    parser.add_argument('--weak_shuffle', dest='weak_shuffle', action='store_true',
                        help="test: use old or new opt.")
    args, unparsed = parser.parse_known_args()

    main(args)