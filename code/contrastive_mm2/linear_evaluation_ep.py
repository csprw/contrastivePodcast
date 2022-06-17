"""
Contrastive multimodal learning: Evaluation
Author: Casper Wortmann
Usage: python evaluate.py
"""
import sys
import os
from collections import defaultdict, Counter

from argparse import ArgumentParser
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib import Path
from train import Cfg
import gc
from time import time
import h5py
import random
from typing import List, Dict, Optional, Union, Tuple, Iterable, Type, Callable

import torch
from torch import nn, Tensor
from torch.utils import data as datautil
from torch.utils.data import Sampler, BatchSampler, DataLoader
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

from transformers import AutoTokenizer, AutoModel, AdamW, AutoConfig
from transformers import get_constant_schedule, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from dacite import from_dict



sys.path.append('../scripts/') 
from train import mmModule, multimodal_loss
from src.data import load_metadata, find_paths, relative_file_path

from omegaconf import OmegaConf
conf = OmegaConf.load("./config.yaml")




###############################################################################
# Move to dataloader.py
class MMloader(object):
    """
    Module that load datasets. 
    """
    def __init__(self, CFG, lin_sep=False, kwargs={}):

        train_dataset_name = CFG.train_dataset
        val_dataset_name = CFG.val_dataset
        test_dataset_name = CFG.test_dataset

        self.batch_size = CFG.batch_size
        self.device = CFG.device
        self.load_full = True if CFG.audio_proj_head in ['gru', 'rnn'] else False
        self.lin_sep = lin_sep

        # Get the datasets
        self.train_dataset, self.train_loader = self.create_dataset(CFG, name=train_dataset_name,  traintest="train")
        self.val_dataset, self.val_loader = self.create_dataset(CFG, name=val_dataset_name,  traintest="val")
        self.test_dataset, self.test_loader = self.create_dataset(CFG, name=test_dataset_name,  traintest="test", shuffle=False)

    def create_dataset(self, CFG, name, traintest, shuffle=True):
        directory = conf.sp_sample_path if name == 'sp_sample' else conf.sp_path
        dataset = self.get_sp_dataset(CFG, directory=directory,  traintest=traintest, load_full=self.load_full, lin_sep=self.lin_sep)

        if (CFG.weak_shuffle and traintest != 'test') or (CFG.ep_level):
            shuffle = False if CFG.ep_level else True

            loader = DataLoader(
                dataset, batch_size=None,  # must be disabled when using samplers
                sampler=BatchSampler(RandomBatchSampler(dataset, self.batch_size, shuffle=shuffle), 
                batch_size=self.batch_size, drop_last=True)
            )
        else:
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)
            loader.collate_fn = dataset.collate_fn
        return dataset, loader

    def get_sp_dataset(self, CFG, directory=conf.sp_sample_path, traintest="train", load_full=False, lin_sep=False):
        if CFG.ep_level:
            # print("ep level")
            dataset = spDatasetEpLevel(CFG, directory=directory, traintest=traintest,  load_full=load_full, lin_sep=lin_sep, device=self.device)
        elif CFG.weak_shuffle and traintest != 'test':
            dataset = spDatasetWeakShuffleLinSep(CFG, directory=directory, traintest=traintest,  load_full=load_full, lin_sep=lin_sep, device=self.device)
        else:
            dataset = spDatasetNoMemory(CFG, directory=directory, traintest=traintest,  load_full=load_full, lin_sep=lin_sep, device=self.device)
        return dataset

class spDatasetNoMemory(datautil.Dataset):
    """
    Spotify Podcast dataset dataloader. 
    """
    def __init__(self, CFG, directory=conf.sp_sample_path, traintest="train", load_full=False, lin_sep=False, device=None):
        
        directory = os.path.join(directory, traintest)
        h5py_files = list(Path(directory).glob('*.h5'))
        print("[spDataset] init from directory ", directory)
        print("[spDataset] found {} h5py files".format(len(h5py_files)))

        self.device = device
        self.lin_sep = lin_sep
        self.load_full = load_full
        self.traintest = traintest
        self.h5py_idx2file = h5py_files

        self.tokenizer = AutoTokenizer.from_pretrained(CFG.text_model_name)
        self.text_max_length = CFG.text_max_length
        
        sample_idx = 0
        idx2file = {}
        self.read_ep2cat()
        
        for h5idx, h5py_file in enumerate(h5py_files):    
            f = h5py.File(h5py_file, 'r')
            for sent_idx in range(len(f['sentences'])):
                
                # Only load data for wich we have a category label
                if lin_sep:
                    episode = f['seg_ts'][sent_idx].decode("utf-8").split('_')[0]
                    if episode not in self.ep2cat.keys():
                        continue
                        
                idx2file[sample_idx] = (h5idx, sent_idx)
                sample_idx += 1
                

                # if sample_idx > 500:
                #     print(" [del4] del del del!!")
                #     break

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

    def read_ep2cat(self):
        ep2cat_path = os.path.join(conf.dataset_path, 'ep2cat.json')
        with open(ep2cat_path) as json_file: 
            self.ep2cat = json.load(json_file)
        ep2cat_map_path = os.path.join(conf.dataset_path, 'ep2cat_mapping.json')
        with open(ep2cat_map_path) as json_file: 
            self.ep2cat_map = json.load(json_file)
        self.num_cats = len(self.ep2cat_map.keys())

    def __len__(self):
        """ Denotes the total number of utterances """
        return len(self.idx2file.keys())

    def __getitem__(self, index):
        """ Return one item from the df """
        if self.traintest == 'test':
            h5py_idx, sent_idx = self.idx2file[index]
            h5py_file = self.h5py_idx2file[h5py_idx]

            f = h5py.File(h5py_file, 'r')
            sent = f['sentences'][sent_idx].decode("utf-8") 

            if self.load_full:
                audio_embeds = torch.Tensor(np.array(f[str(sent_idx)]))
            else:
                audio_embeds = torch.Tensor(f['mean_embeddings'][sent_idx])
            target = f['seg_ts'][sent_idx].decode("utf-8") 

            if self.lin_sep:
                episode = target.split('_')[0]
                cat = self.ep2cat[episode]
            sample = (sent, audio_embeds, target, cat)

        else:
            h5py_idx, sent_idx = self.idx2file[index]
            h5py_file = self.h5py_idx2file[h5py_idx]

            f = h5py.File(h5py_file, 'r')
            sent = f['sentences'][sent_idx].decode("utf-8")

            if self.load_full:
                audio_embeds = torch.Tensor(np.array(f[str(sent_idx)]))
            else:
                audio_embeds = torch.Tensor(f['mean_embeddings'][sent_idx])
            sample = (sent, audio_embeds)
        
        f.close()
        return sample

    def full_batching_collate(self, batch):
        """ Return a batch containing samples of the SP dataset"""
        text_embeds = []
        audio_embeds = []
        full_text = []
        lengths  = []
        cats = []
        
        for example in batch:
            full_text.append(example[0])
            text_embeds.append(example[0])
            audio_embeds.append(example[1])
            lengths.append(len(example[1]))
            cats.append(example[3])
        
        # Pad the audio embeddings
        padded_audio_embeds = pad_sequence(audio_embeds, batch_first=True).to(self.device)
        
        # Tokenize text
        text_embeds = self.tokenizer(
            text_embeds, padding=True, truncation=True, max_length=self.text_max_length, return_tensors='pt'
        ).to(self.device)

        if self.traintest == 'test':
            targs = [example[2] for example in batch]
            cats = torch.tensor(cats).to(self.device)
            return text_embeds, padded_audio_embeds, lengths, targs, full_text, cats
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
        full_text = []
        cats =[]
        
        for example in batch:
            full_text.append(example[0])
            text_embeds.append(example[0])
            audio_embeds.append(example[1])
            cats.append(example[3])

        audio_embeds = torch.stack(audio_embeds).to(self.device)

        # Tokenize text
        max_length = 32 # TODO: is dit nodig?
        text_embeds = self.tokenizer(
            text_embeds, padding=True, truncation=True, max_length=max_length, return_tensors='pt'
        ).to(self.device)
        
        # return text_embeds, audio_embeds, lengths
        if self.traintest == 'test':
            targs = [example[2] for example in batch]
            cats = torch.tensor(cats).to(self.device)
            return text_embeds, audio_embeds, lengths, targs, full_text, cats
        else:
            return text_embeds, audio_embeds, lengths

class spDatasetWeakShuffleLinSep(datautil.Dataset):
    """
    Spotify Podcast dataset dataloader. 
    Weak shuffling to enable shuffle while clustering sentences of similar length.
    """
    def __init__(self, CFG, directory=conf.sp_sample_path, traintest="train", load_full=False,lin_sep=False, device=None):
        
        directory = os.path.join(directory, traintest)
        h5py_files = list(Path(directory).glob('*.h5'))
        print("[spDataset] init from directory [WEAKSHUFFLE] ", directory)
        print("[spDataset] found {} h5py files".format(len(h5py_files)))

        self.device = device
        self.lin_sep = lin_sep
        self.read_ep2cat()
        
        idx2file = {}
        sample_idx = 0
        self.h5py_idx2file = h5py_files
        self.load_full = load_full
        self.traintest = traintest

        self.tokenizer = AutoTokenizer.from_pretrained(CFG.text_model_name)
        self.text_max_length = CFG.text_max_length
        self.file_startstop = []

        for h5idx, h5py_file in enumerate(h5py_files):    
            f = h5py.File(h5py_file, 'r')
            print("[spdataset] loading {}/{}: {}".format(h5idx, len(h5py_files), h5py_file))
            start_idx = sample_idx

            for sent_idx in range(len(f['sentences'])):
                
                # Only load data for wich we have a category label
                if lin_sep:
                    episode = f['seg_ts'][sent_idx].decode("utf-8").split('_')[0]
                    if episode not in self.ep2cat.keys():
                        continue

                idx2file[sample_idx] = (h5idx, sent_idx)
                sample_idx += 1

                if CFG.max_train_samples > 0 and traintest == 'train' and sample_idx >= CFG.max_train_samples:
                # if CFG.max_train_samples > 0 and sample_idx >= CFG.max_train_samples:
                    print("[del] Max exceeded {}".format(sample_idx))
                    f.close()
                    self.file_startstop.append((start_idx, sample_idx))
                    break
            else:
                f.close()
                self.file_startstop.append((start_idx, sample_idx))
                continue
            break

        self.idx2file = idx2file
        self.last_idx = -1
        self.mean_embeds = None
        self.f =  h5py.File(h5py_file, 'r')

    def read_ep2cat(self):
        ep2cat_path = os.path.join(conf.dataset_path, 'ep2cat.json')
        with open(ep2cat_path) as json_file: 
            self.ep2cat = json.load(json_file)
        ep2cat_map_path = os.path.join(conf.dataset_path, 'ep2cat_mapping.json')
        with open(ep2cat_map_path) as json_file: 
            self.ep2cat_map = json.load(json_file)
        self.num_cats = len(self.ep2cat_map.keys())

    def __len__(self):
        """ Denotes the total number of utterances """
        return len(self.idx2file.keys())

    def __getitem__(self, index):
        """ Return one item from the df """
        if self.traintest == 'test':
            print("Weak shuffling not supported for test set")
            raise NotImplementedError

        else:
            text_embeds = []
            audio_embeds = []
            full_text = []
            lengths  = []
            cats =[]

            last_idx = self.last_idx
            for enum, i in enumerate(index):
                h5py_idx, sent_idx = self.idx2file[i]
                if h5py_idx != last_idx:
                    # Clear memory
                    self.f.close()
                    if not self.load_full:
                        del self.mean_embeds
                    gc.collect()
                    h5py_file = self.h5py_idx2file[h5py_idx]
                    self.f = h5py.File(h5py_file, 'r')
                    print("[del1] loaded new h5py file: ", h5py_idx, h5py_file)
                    if not self.load_full:
                        self.mean_embeds = torch.Tensor(np.array(self.f['mean_embeddings']))
                
                sent = self.f['sentences'][sent_idx].decode("utf-8")
                if self.load_full:
                    audio_embed = torch.Tensor(np.array(self.f[str(sent_idx)]))
                    lengths.append(len(audio_embed))
                else:
                    audio_embed = self.mean_embeds[sent_idx]

                full_text.append(sent)
                text_embeds.append(sent)
                audio_embeds.append(audio_embed)
                
                if self.lin_sep:
                    episode = self.f['seg_ts'][sent_idx].decode("utf-8").split('_')[0]
                    cats.append(self.ep2cat[episode])
                last_idx = h5py_idx

            self.last_idx = last_idx
            cats = torch.tensor(cats).to(self.device)

             # Pad the audio embeddings
            audio_embeds = pad_sequence(audio_embeds, batch_first=True).to(self.device)
            
            # Tokenize text
            text_embeds = self.tokenizer(
                text_embeds, padding=True, truncation=True, max_length=self.text_max_length, return_tensors='pt'
            ).to(self.device)

        if self.lin_sep:    # TODO: always return cats
            return text_embeds, audio_embeds, lengths, cats
        else:
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
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        # self.batch_ids = torch.randperm(int(self.n_batches)) # sorted

        # Not shuffled:
        if shuffle:
            self.batch_ids = self.shuffle_within_file(dataset)
        else:
            self.batch_ids = torch.arange(int(self.n_batches))
        # self.batch_ids = torch.arange(int(self.n_batches))
        # self.batch_ids = self.shuffle_within_file(dataset)

    def shuffle_within_file(self, dataset):
        batch_ids = np.arange(int(self.n_batches))
        file_startstop = [(int(i[0]/self.batch_size), int(i[1]/ self.batch_size)) for i in dataset.file_startstop]
        blocks = [list(batch_ids[i[0]:i[1]]) for i in file_startstop]
        blocks = [random.sample(b, len(b)) for b in blocks]
        batch_ids[:] = [b for bs in blocks for b in bs]
        return torch.tensor(batch_ids)

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for id in self.batch_ids:
            idx = torch.arange(id * self.batch_size, (id + 1) * self.batch_size)
            # TODO: only do this if in the same file
            # t = torch.randperm(idx.shape[0])
            # idx= idx[t].view(idx.size())
            for index in idx:
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(int(self.n_batches) * self.batch_size, self.dataset_length)
            for index in idx:
                yield int(index)


class spDatasetEpLevel(datautil.Dataset):
    """
    Spotify Podcast dataset dataloader. 
    Weak shuffling to enable shuffle while clustering sentences of similar length.
    """
    def __init__(self, CFG, directory=conf.sp_sample_path, traintest="train", load_full=False,lin_sep=False, device=None):
        
        directory = os.path.join(directory, traintest)
        h5py_files = list(Path(directory).glob('*.h5'))
        print("[spDataset] init from directory [EpLevel] ", directory)
        print("[spDataset] found {} h5py files".format(len(h5py_files)))

        self.device = device
        self.lin_sep = lin_sep
        self.read_ep2cat()
        
        ### DELETE [del4]
        # CFG.max_train_samples = 128 * 1
        self.return_targets = True
        
        idx2file = {}
        sample_idx = 0
        self.h5py_idx2file = h5py_files
        self.load_full = load_full
        self.traintest = traintest

        self.tokenizer = AutoTokenizer.from_pretrained(CFG.text_model_name)
        self.text_max_length = CFG.text_max_length
        self.file_startstop = []
        self.target_idx = defaultdict(list)

        for h5idx, h5py_file in enumerate(h5py_files):    
            f = h5py.File(h5py_file, 'r')
            print("[spdataset] loading {}/{}: {}".format(h5idx, len(h5py_files), h5py_file))
            start_idx = sample_idx

            for sent_idx in range(len(f['sentences'])):
                
                # Only load data for wich we have a category label
                if lin_sep:
                    episode = f['seg_ts'][sent_idx].decode("utf-8").split('_')[0]
                    if episode not in self.ep2cat.keys():
                        # print("[del4] continue")
                        continue
                    
                    self.target_idx[episode].append((h5idx, sent_idx))

                idx2file[sample_idx] = (h5idx, sent_idx)
                sample_idx += 1

                # if CFG.max_train_samples > 0 and traintest == 'train' and sample_idx >= CFG.max_train_samples:
                if CFG.max_train_samples > 0  and sample_idx >= CFG.max_train_samples:
                    print("[del] Max exceeded {}".format(sample_idx))
                    f.close()
                    self.file_startstop.append((start_idx, sample_idx))
                    break
                #elif sample_idx > 5000000:
                # elif sample_idx > 500000:
                elif sample_idx > 500:
                    f.close()
                    self.file_startstop.append((start_idx, sample_idx))
                    print("[del] Max exceeded {}".format(sample_idx))
                    break
            else:
                f.close()
                self.file_startstop.append((start_idx, sample_idx))
                continue
            break

        self.idx2targetfile = {}
        c = 0
        for k, v in self.target_idx.items():
            target = k
            for tup in v:
                h5idx = tup[0]
                sent_idx = tup[1]
                self.idx2targetfile[c] = (h5idx, sent_idx, target)
                c+= 1
        self.idx2file = idx2file
        self.last_idx = -1
        self.mean_embeds = None
        self.f =  h5py.File(h5py_file, 'r')

    def read_ep2cat(self):
        ep2cat_path = os.path.join(conf.dataset_path, 'ep2cat.json')
        with open(ep2cat_path) as json_file: 
            self.ep2cat = json.load(json_file)
        ep2cat_map_path = os.path.join(conf.dataset_path, 'ep2cat_mapping.json')
        with open(ep2cat_map_path) as json_file: 
            self.ep2cat_map = json.load(json_file)
        self.num_cats = len(self.ep2cat_map.keys())
        print("Number of cats: ", self.num_cats)

    def __len__(self):
        """ Denotes the total number of utterances """
        return len(self.idx2file.keys())

    def __getitem__(self, index):
        """ Return one item from the df """
        if self.return_targets:
            text_embeds = []
            audio_embeds = []
            full_text = []
            lengths  = []
            cats = []
            targs = []

            last_idx = self.last_idx
            for enum, i in enumerate(index):
                h5py_idx, sent_idx, episode = self.idx2targetfile[i]
                if h5py_idx != last_idx:
                    # Clear memory
                    self.f.close()
                    if not self.load_full:
                        del self.mean_embeds
                    gc.collect()
                    h5py_file = self.h5py_idx2file[h5py_idx]
                    self.f = h5py.File(h5py_file, 'r')
                    print("[del2] loaded new h5py file: ", h5py_idx, h5py_file)
                    if not self.load_full:
                        self.mean_embeds = torch.Tensor(np.array(self.f['mean_embeddings']))
                
                sent = self.f['sentences'][sent_idx].decode("utf-8")
                if self.load_full:
                    audio_embed = torch.Tensor(np.array(self.f[str(sent_idx)]))
                    lengths.append(len(audio_embed))
                else:
                    audio_embed = self.mean_embeds[sent_idx]

                full_text.append(sent)
                text_embeds.append(sent)
                audio_embeds.append(audio_embed)
                
                if self.lin_sep or self.return_targets:
                    episode = self.f['seg_ts'][sent_idx].decode("utf-8").split('_')[0]
                    cats.append(self.ep2cat[episode])
                    targs.append(episode)
                last_idx = h5py_idx

            self.last_idx = last_idx
            cats = torch.tensor(cats).to(self.device)

             # Pad the audio embeddings
            audio_embeds = pad_sequence(audio_embeds, batch_first=True).to(self.device)
            
            # Tokenize text
            text_embeds = self.tokenizer(
                text_embeds, padding=True, truncation=True, max_length=self.text_max_length, return_tensors='pt'
            ).to(self.device)


            return text_embeds, audio_embeds, lengths, cats, targs


        else:
            text_embeds = []
            audio_embeds = []
            full_text = []
            lengths  = []
            cats =[]

            last_idx = self.last_idx
            for enum, i in enumerate(index):
                h5py_idx, sent_idx = self.idx2file[i]
                if h5py_idx != last_idx:
                    # Clear memory
                    self.f.close()
                    if not self.load_full:
                        del self.mean_embeds
                    gc.collect()
                    h5py_file = self.h5py_idx2file[h5py_idx]
                    self.f = h5py.File(h5py_file, 'r')
                    print("[del3] loaded new h5py file: ", h5py_idx, h5py_file)
                    if not self.load_full:
                        self.mean_embeds = torch.Tensor(np.array(self.f['mean_embeddings']))
                
                sent = self.f['sentences'][sent_idx].decode("utf-8")
                if self.load_full:
                    audio_embed = torch.Tensor(np.array(self.f[str(sent_idx)]))
                    lengths.append(len(audio_embed))
                else:
                    audio_embed = self.mean_embeds[sent_idx]

                full_text.append(sent)
                text_embeds.append(sent)
                audio_embeds.append(audio_embed)
                
                if self.lin_sep:
                    episode = self.f['seg_ts'][sent_idx].decode("utf-8").split('_')[0]
                    cats.append(self.ep2cat[episode])
                last_idx = h5py_idx

            self.last_idx = last_idx
            cats = torch.tensor(cats).to(self.device)

             # Pad the audio embeddings
            audio_embeds = pad_sequence(audio_embeds, batch_first=True).to(self.device)
            
            # Tokenize text
            text_embeds = self.tokenizer(
                text_embeds, padding=True, truncation=True, max_length=self.text_max_length, return_tensors='pt'
            ).to(self.device)

        if self.lin_sep:    # TODO: always return cats
            return text_embeds, audio_embeds, lengths, cats
        else:
            return text_embeds, audio_embeds, lengths

####### UTILS
def freeze_network(model):
    for name, p in model.named_parameters():
        p.requires_grad = False

###### Linear Evaluation
# class LinearEvaluationModel(nn.Module):
#     """
#         This loss expects as input a batch consisting of ... etc
#     """
#     def __init__(self, full_model, CFG, data_loader, modality='text'):
#         """
#         test
#         """
#         super(LinearEvaluationModel, self).__init__()
#         self.text_model = full_model.text_model
#         self.audio_model = full_model.audio_model

#         self.device = torch.device(CFG.device)
#         self.batch_size = full_model.batch_size
        
#         freeze_network(self.text_model)
#         freeze_network(self.audio_model)

#         self.hidden_dim = CFG.final_projection_dim
#         self.output_dim = data_loader.train_dataset.num_cats

#         if not args.mlp:
#             self.projectionhead = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim))
#         else:
#             self.projectionhead = nn.Sequential(
#                nn.Linear(self.hidden_dim, self.hidden_dim),
#                nn.ReLU(),
#                nn.Linear(self.hidden_dim, self.output_dim),
#             )
#         self.criterion = nn.CrossEntropyLoss()
#         self.modality = modality

#         self.projectionhead.to(self.device)
#         self.text_model.to(self.device)
#         self.audio_model.to(self.device)

        
#     def forward(self, sentence_features: Iterable[Dict[str, Tensor]], audio_features: Tensor, seq_len, cats):
#         with torch.no_grad():

#             # Encode features
#             reps_text = self.text_model(sentence_features)['sentence_embedding']
#             reps_audio = self.audio_model((audio_features, seq_len))

#             # Normalise features
#             reps_audio = reps_audio / reps_audio.norm(dim=1, keepdim=True)
#             reps_text = reps_text / reps_text.norm(dim=1, keepdim=True)
        
#         if self.modality == 'text':
#             preds = self.projectionhead(reps_text)
#         else:
#             preds = self.projectionhead(reps_audio)

#         loss = self.criterion(preds, cats)
#         metrics = self.get_metrics(preds.detach().cpu(), cats.detach().cpu())
#         return loss, metrics
        
#     def get_metrics(self, preds, targets):
#         metrics = {}
#         y_pred = torch.argmax(preds, axis=1)
        
#         # print("\t\t\t y_pred: ", torch.unique(y_pred), torch.bincount(targets))
        
#         metrics['acc'] = torch.sum(y_pred == targets) / targets.shape[0]
#         metrics['targets'] = targets.tolist()
#         metrics['preds'] = y_pred.tolist()
        
#         return metrics

    
# class LinearEvalator(nn.Module):
#     """
#         This loss expects as input a batch consisting of ... etc
#     """
#     def __init__(self, lin_eval_model, CFG, data_loader, modality='text'):
#         """
#         test
#         """
#         super(LinearEvalator, self).__init__()
#         self.modality=modality
#         self.lin_eval_model = lin_eval_model
        
#         self.data_loader = data_loader
#         self.output_path = CFG.log_name

#         # to config file
#         self.lin_max_epochs = args.num_epochs
#         self.in_batch_size = 256
#         lin_lr = 0.001
#         lin_weight_decay = 1.0e-6
        
#         self.optimizer = torch.optim.Adam(
#             self.lin_eval_model.parameters(),
#             lr=lin_lr,
#             weight_decay=lin_weight_decay,
#         )
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer,
#             mode="min",
#             factor=0.1,
#             patience=5,
#             threshold=0.0001,
#             threshold_mode="rel",
#             cooldown=0,
#             min_lr=0,
#             eps=1e-08,
#             verbose=False,
#         )

        
#     def fit(self):
#         #self.lin_eval_model.projectionhead.train()
#         self.acc_per_epoch = []
#         for epoch in range(self.lin_max_epochs):
#             print("-- -- -- Epoch: ", epoch)
#             accs =[]
#             for step, batch in enumerate(iter(self.data_loader.train_loader)):
#                 sent_features, audio_features, seq_len, cats = batch
#                 self.optimizer.zero_grad()

#                 loss, metrics = self.lin_eval_model(sent_features, audio_features, seq_len, cats)    
#                 accs.append(metrics['acc'])

#                 loss.backward()
#                 self.optimizer.step()
#                 self.scheduler.step(loss)
                
#                 if step % 100 == 0:
#                     print("Loss: {} \t acc: {}".format(loss, metrics['acc']))

#             print("-- Train epoch Mean acc: ", np.mean(accs))
#             self.acc_per_epoch.append(np.mean(accs))

#         print("Train done, accs per epoch: ", self.acc_per_epoch)

#         # TODO: for now save intermediate, in the end only final round.
#         self.evaluate()
#         self.save_results(epoch)
        
#     def evaluate(self):
#         accs = []
#         self.lin_eval_model.projectionhead.eval()
#         preds = []
#         targets = []
        
#         print("-- -- -- Start evaluation: ")
#         with torch.no_grad():
#             for step, batch in enumerate(iter(self.data_loader.test_loader)):
#                 sent_features, audio_features, seq_len, targs, _, cats = batch
#                 #self.optimizer.zero_grad()

#                 loss, metrics = self.lin_eval_model(sent_features, audio_features, seq_len, cats)    
#                 accs.append(metrics['acc'])
#                 print("eval Loss: {} \t acc: {}".format(loss, metrics['acc']))
                
#                 preds.extend(metrics['preds'])
#                 targets.extend(metrics['targets'])

#         self.eval_mean_acc = np.mean(accs)
#         self.preds = preds
#         self.targets = targets
        
#     def save_results(self, epoch=0):
#         # Save results to csv
#         lin_eval_res = {'eval_acc': self.eval_mean_acc,
#                        'acc_per_epoch': list(self.acc_per_epoch)}
#         out = os.path.join(self.output_path, 
#             '{}_{}_linear_evaluation_results.csv'.format(epoch, self.modality))
#         df = pd.DataFrame(lin_eval_res)
#         df.T.to_csv(out, sep=';')
#         print(df.T.to_string())

#         # Save results as confusion matrix
#         classes = list(self.data_loader.test_dataset.ep2cat_map.keys())
#         self.make_cm(self.preds, self.targets, classes, epoch)

#     def make_cm(self, y_pred, y_true, classes, epoch=0):
#         # Build confusion matrix
#         print("[del4] this is make_cm: ", y_pred)
#         print("And classes: ", classes)
#         cf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
#         df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index = [i for i in classes],
#             columns = [i for i in classes])
#         plt.figure(figsize = (12,7))
#         sn.heatmap(df_cm, annot=True, cmap='Blues')
        
#         out = os.path.join(self.output_path, 
#             '{}_{}_linear_evaluation_cm.png'.format(epoch, self.modality))
#         print("Saving to: ", out)
#         plt.savefig(out)
#         # plt.show()


class epLevelCreator(nn.Module):
    """
        This loss expects as input a batch consisting of ... etc
    """
    def __init__(self, full_model, CFG, data_loader):
        """
        test
        """
        super(epLevelCreator, self).__init__()
        self.text_model = full_model.text_model
        self.audio_model = full_model.audio_model

        self.device = torch.device(CFG.device)
        self.batch_size = full_model.batch_size
        
        freeze_network(self.text_model)
        freeze_network(self.audio_model)
        
        self.data_loader = data_loader
        self.output_path = CFG.log_name

        self.save_intermediate = os.path.join(conf.data_path, os.path.split(CFG.log_name)[-1], "len_eval.h5")
        # self.f = h5py.File(self.save_intermediate, "w")  
        self.means = defaultdict(dict)
        
        self.cur_ep = None
        self.mean_cats = []
        self.mean_a_embeds = []
        self.mean_t_embeds = []
        self.processed_eps = []

        self.cur_a_embeds = []
        self.cur_t_embeds = []

    def create(self):

        print("-------- Creating embeddings")
        with torch.no_grad():
            for step, batch in enumerate(iter(self.data_loader.train_loader)):
                sent_features, audio_features, seq_len, cats, targets = batch

                # Encode features
                reps_text = self.text_model(sent_features)['sentence_embedding']
                reps_audio = self.audio_model((audio_features, seq_len))

                # Normalise features
                reps_audio = reps_audio / reps_audio.norm(dim=1, keepdim=True)
                reps_text = reps_text / reps_text.norm(dim=1, keepdim=True)

                num_targs = len(list(set(targets)))
                if self.cur_ep == None:
                    self.cur_ep = targets[0]

                if num_targs > 1 or self.cur_ep != targets[0]:
                    if num_targs > 1:
                        print("OPT 0: Num targs > 1: ",num_targs )
                    else:
                        print("OPT 1: entrie batch different")
                    for idx, targ in enumerate(targets):
                        if self.cur_ep != targ:
                            if targ in self.processed_eps:
                                print("DEBUG: already exists? should not be possible")
                                raise
                            self.mean_cats.append(cats[idx].item())


                            self.mean_a_embeds.append(torch.mean(torch.stack(self.cur_a_embeds, dim=0), dim=0).detach().cpu())
                            self.mean_t_embeds.append(torch.mean(torch.stack(self.cur_t_embeds, dim=0), dim=0).detach().cpu())
                            self.processed_eps.append(targ.cpu())
                            self.cur_ep = targ

                        self.cur_a_embeds.append(reps_audio[idx])
                        self.cur_t_embeds.append(reps_text[idx])

                else:
                    print("OPT 2: append")
                    self.cur_a_embeds.append(torch.mean(reps_audio, dim=0))
                    self.cur_t_embeds.append(torch.mean(reps_text, dim=0))

### zometeen naar boven
class epDataset(datautil.Dataset):
    def __init__(self, CFG, eplevel):
        self.mean_cats = eplevel.mean_cats
        self.mean_a_embeds = eplevel.mean_a_embeds
        self.mean_t_embeds = eplevel.mean_t_embeds

        self.device = CFG.device

        self.read_ep2cat()

    def read_ep2cat(self):
        ep2cat_path = os.path.join(conf.dataset_path, 'ep2cat.json')
        with open(ep2cat_path) as json_file: 
            self.ep2cat = json.load(json_file)
        ep2cat_map_path = os.path.join(conf.dataset_path, 'ep2cat_mapping.json')
        with open(ep2cat_map_path) as json_file: 
            self.ep2cat_map = json.load(json_file)
        self.num_cats = len(self.ep2cat_map.keys())

    def __len__(self):
        return len(self.mean_cats)

    def __getitem__(self, idx):
        a_embeds = self.mean_a_embeds[idx]
        t_embeds = self.mean_t_embeds[idx]
        targets =  self.mean_cats[idx]
        return (a_embeds, t_embeds, targets)

    
    # def batching_collate(self, batch):
    #     """ Return a batch containing samples of the SP dataset"""
    #     print("[del4] in batching collate!")
    #     # print("This is batch: ", batch)
    #     print(len(batch))
    #     print(type(batch[0]))
    #     print(type(batch[1]))
    #     print(type(batch[2]))
    #     a_embeds, t_embeds, targets = batch

    #     return  a_embeds.to(self.device), t_embeds.to(self.device), torch.tensor(targets).to(self.device)






class LinearEvaluatorEplevel(nn.Module):
    """
        This loss expects as input a batch consisting of ... etc
    """
    def __init__(self, CFG, train_ep_loader, classes, modality='text'):
        """
        test
        """
        super(LinearEvaluatorEplevel, self).__init__()
        self.modality=modality
        self.device = CFG.device
        self.train_ep_loader = train_ep_loader
        self.val_ep_loader = train_ep_loader
        self.output_path = CFG.log_name

        # to config file
        self.lin_max_epochs = args.num_epochs
        # self.lin_max_epochs = 500

        lin_lr = 0.01
        lin_weight_decay = 1.0e-6

        self.classes = classes
        self.hidden_dim = CFG.final_projection_dim
        self.output_dim = len(classes)

        self.embed_norm = True

        mlp = args.mlp
        if not mlp:
            self.projectionhead = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim))
        else:
            self.projectionhead = nn.Sequential(
                nn.BatchNorm1d(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )

        self.criterion = nn.CrossEntropyLoss()
        self.projectionhead.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.projectionhead.parameters(),
            lr=lin_lr,
            weight_decay=lin_weight_decay,
        )

        
    def fit(self):
        #self.lin_eval_model.projectionhead.train()
        self.acc_per_epoch = []
        for epoch in range(self.lin_max_epochs):
            print("-- -- -- Epoch: ", epoch)
            self.projectionhead.train()
            self.train()
            accs =[]
            for step, batch in enumerate(iter(self.train_ep_loader)):
                # sent_features, audio_features, seq_len, cats = batch
                mean_a_embeds, mean_t_embeds, cats = batch
                mean_a_embeds = mean_a_embeds.to(self.device)
                mean_t_embeds = mean_t_embeds.to(self.device)
                cats = cats.to(self.device)

                if self.embed_norm:
                    mean_a_embeds = mean_a_embeds / (torch.sum(mean_a_embeds, 1).unsqueeze(-1))
                    mean_t_embeds = mean_t_embeds / (torch.sum(mean_t_embeds, 1).unsqueeze(-1))

                if self.modality == 'text':
                    output = self.projectionhead(mean_t_embeds)
                else:
                    output = self.projectionhead(mean_a_embeds)

                # Forward pass
                self.optimizer.zero_grad()
                loss = self.criterion(output, cats)

                loss.backward()

                self.optimizer.step()

                y_pred = torch.argmax(output, axis=1)

                
                metrics = self.get_metrics(output.detach().cpu(), cats.detach().cpu())
                accs.append(metrics['acc'])
                if torch.equal(y_pred.detach().cpu(), cats.detach().cpu()):
                    print("SAMMEEE ", step, step)
                    raise

                # if step > 0:
                #     break

                if step % 100 == 0:
                    print("___________________________________________________")
                    print("output:" , epoch, step, loss.item(), (y_pred[:10]), (cats[:10]))
                    print(Counter(cats.detach().cpu().tolist()), Counter(y_pred.detach().cpu().tolist()))
                    # print("Loss: {} \t acc: {}".format(loss, metrics['acc']))

            print("-- Train epoch Mean acc: ", np.mean(accs))
            self.acc_per_epoch.append(np.mean(accs))
            # TODO: for now save intermediate, in the end only final round.
            self.evaluate()
            self.save_results(epoch)
        
    def evaluate(self):
        accs = []
        self.projectionhead.eval()
        full_preds = []
        full_targets = []
        
        print("-- -- -- Start evaluation ")
        self.projectionhead.eval()
        with torch.no_grad():
            for step, batch in enumerate(iter(self.val_ep_loader)):
                
                # sent_features, audio_features, seq_len, targs, _, cats = batch
                mean_a_embeds, mean_t_embeds, cats = batch
                
                if self.embed_norm:
                    mean_a_embeds = mean_a_embeds / (torch.sum(mean_a_embeds, 1).unsqueeze(-1))
                    mean_t_embeds = mean_t_embeds / (torch.sum(mean_t_embeds, 1).unsqueeze(-1))

                if self.modality == 'text':
                    output = self.projectionhead(mean_t_embeds)
                else:
                    output = self.projectionhead(mean_a_embeds)

                
                metrics = self.get_metrics(output.detach().cpu(), cats.detach().cpu())
                accs.append(metrics['acc'])
                full_preds.extend(metrics['preds'])
                full_targets.extend(metrics['targets'])
                
                # y_pred = torch.argmax(output, axis=1)
                #print("eval step: ", step, (y_pred[:10]), (cats[:10]))
                
                if step > 1000: #TODO 
                    break

        self.eval_mean_acc = np.mean(accs)
        self.preds = full_preds
        self.targets = full_targets
        print("Evaluaction accuracy: ", self.eval_mean_acc)
        self.projectionhead.train()

    def get_metrics(self, preds, targets):
        metrics = {}
        y_pred = torch.argmax(preds, axis=1)
        # print("\t\t\t y_pred: ", torch.unique(y_pred), torch.bincount(targets))
        metrics['acc'] = torch.sum(y_pred == targets) / targets.shape[0]
        metrics['targets'] = targets.tolist()
        metrics['preds'] = y_pred.tolist()
        
        return metrics
        
    def save_results(self, epoch=0):
        # Save results to csv
        lin_eval_res = {'eval_acc': self.eval_mean_acc,
                       'acc_per_epoch': list(self.acc_per_epoch)}
        out = os.path.join(self.output_path, 
            'ep_{}_linear_evaluation_results.csv'.format(self.modality))
        df = pd.DataFrame(lin_eval_res)
        # df.T.to_csv(out, sep=';')
        print(df.T.to_string())

        # Save results as confusion matrix
        classes = self.classes
        self.make_cm(self.preds, self.targets, classes, epoch)

    def make_cm(self, y_pred, y_true, classes, epoch=0):
        # Build confusion matrix
        print("[del4] this is make_cm: ",Counter(y_true), Counter(y_pred))
        # print("And classes: ", classes)
        cf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)), normalize=None)
        df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
            columns = [i for i in classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True, cmap='Blues')
        
        out = os.path.join(self.output_path, 
            'ep_{}_{}_linear_evaluation_cm.png'.format(epoch, self.modality))
        print("Saving to: ", out)
        plt.savefig(out)
        # plt.show()

def main(args):
    model_path = args.model_path
    print("[Load model] from ", model_path)
    model_weights_path = os.path.join(model_path, "output/full_model_weights.pth")
    model_config_path = os.path.join(model_path, 'config.json')

    # Opening JSON file
    f = open(model_config_path)
    model_config = json.load(f)
    fullcfg = from_dict(data_class=Cfg, data=model_config)
    fullcfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    fullcfg.ep_level = args.ep_level
    fullcfg.weak_shuffle = False            # For evaluation turn shuffling off
    ### DELETE [del4]
    # fullcfg.max_train_samples = 128 * 20
    print("[Load model] config loaded: ", fullcfg)

    # Create dataloader
    data_loader = MMloader(fullcfg, lin_sep=True)

    # Load the model
    full_model = mmModule(fullcfg)
    full_model.load_state_dict(torch.load(model_weights_path,  map_location=fullcfg.device))              
    full_model = full_model.to(fullcfg.device)     
    full_model.eval()

    # Create representations on epiode level
    eplevel = epLevelCreator(full_model, fullcfg, data_loader)
    eplevel.create()


    ep_dataset= epDataset(fullcfg, eplevel)
    ep_loader = DataLoader(ep_dataset, batch_size=args.lin_batch_size, shuffle=False, drop_last=True)

    # Calculate results for text
    classes = list(ep_dataset.ep2cat_map.keys())
    evaluator = LinearEvaluatorEplevel(fullcfg, ep_loader, classes, modality="text")
    evaluator.fit()

    # Calculate results for audio
    # classes = list(ep_dataset.ep2cat_map.keys())
    # evaluator = LinearEvaluatorEplevel(fullcfg, ep_loader, classes, modality="audio")
    # evaluator.fit()



if __name__ == "__main__":
    print("[evaluate] reading data from: ")
    print(conf.dataset_path)

    # Parse flags in command line arguments
    parser = ArgumentParser()

    parser.add_argument('--model_path', type=str, default="logs/run2-clip_loss_None_sph_768_False_2022-06-12_12-44-00",
                        help='Folder where model weights are saved.')

    parser.add_argument('--lin_batch_size', type=int, default=256,
                        help='Lineaer evaluation batchsize.')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs to train. ')
    parser.add_argument('--mlp', action='store_true', default=False,
                    help='Whether to use multiple layers.')
    parser.add_argument('--ep_level', action='store_true', default=False,
                    help='Whether to train on episode level.')
    args, unparsed = parser.parse_known_args()

    main(args)
