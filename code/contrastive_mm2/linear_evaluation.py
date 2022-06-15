"""
Contrastive multimodal learning: Evaluation
Author: Casper Wortmann
Usage: python evaluate.py
"""
import sys
import os

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
            self.train_dataset = self.get_sp_dataset(CFG, directory=conf.sp_sample_path, traintest="train", load_full=self.load_full, lin_sep=lin_sep, device=self.device)
        elif train_dataset_name == "sp":
            self.train_dataset = self.get_sp_dataset(CFG, directory=conf.sp_path,  traintest="train", load_full=self.load_full, lin_sep=lin_sep, device=self.device)
        else:
            raise Exception('Unknown dataset')
        if CFG.weak_shuffle:
            self.train_loader = DataLoader(
                self.train_dataset, batch_size=None,  # must be disabled when using samplers
                sampler=BatchSampler(RandomBatchSampler(self.train_dataset, batch_size), 
                batch_size=batch_size, drop_last=True)
            )
        else:
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
            self.train_loader.collate_fn = self.train_dataset.collate_fn
        print("[MMloader] train dataset loaded, length: ", len(self.train_dataset))

        if val_dataset_name == "sp_sample":
            self.val_dataset = self.get_sp_dataset(CFG, directory=conf.sp_sample_path,  traintest="val", load_full=self.load_full, lin_sep=lin_sep, device=self.device)
        elif val_dataset_name == "sp":
            self.val_dataset  = self.get_sp_dataset(CFG, directory=conf.sp_path,  traintest="val",  load_full=self.load_full, lin_sep=lin_sep, device=self.device)
        else:
            raise Exception('Unknown dataset')
        if CFG.weak_shuffle:
            self.val_loader = DataLoader(
                self.val_dataset, batch_size=None,  # must be disabled when using samplers
                sampler=BatchSampler(RandomBatchSampler(self.val_dataset , batch_size), batch_size=batch_size, drop_last=True)
            )
        else:
            self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
            self.val_dataset = self.val_dataset
            self.val_loader.collate_fn = self.val_dataset.collate_fn
        print("[MMloader] val dataset loaded, length: ", len(self.val_dataset))

        if test_dataset_name == "sp_sample":
            test_dataset = self.get_sp_dataset(CFG, directory=conf.sp_sample_path,  traintest="test", load_full=self.load_full, lin_sep=lin_sep, device=self.device)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs) 
        elif test_dataset_name == "sp":
            test_dataset = self.get_sp_dataset(CFG, directory=conf.sp_path,  traintest="test",  load_full=self.load_full, lin_sep=lin_sep, device=self.device)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
        else:
            raise Exception('Unknown dataset')
        self.test_dataset = test_dataset
        self.test_loader.collate_fn = self.test_dataset.collate_fn

        
        print("[MMloader] test dataset loaded, length: ", len(test_dataset))

    def get_sp_dataset(self, CFG, directory=conf.sp_sample_path, traintest="train", load_full=False, lin_sep=False, device=None):
        if CFG.weak_shuffle and traintest != 'test':
            dataset = spDatasetWeakShuffleLinSep(CFG, directory=directory, traintest=traintest,  load_full=load_full, lin_sep=lin_sep, device=device)
        else:
            dataset = spDatasetNoMemory(CFG, directory=directory, traintest=traintest,  load_full=load_full, lin_sep=lin_sep, device=device)
        
        return dataset

class spDatasetNoMemory(datautil.Dataset):
    """
    Spotify Podcast dataset dataloader. 
    """
    def __init__(self, CFG, directory=conf.sp_sample_path, traintest="train", load_full=False, lin_sep=False, device=None):
        print("[spDataset] init from directory ", directory, traintest)
        directory = os.path.join(directory, traintest)
        h5py_files = list(Path(directory).glob('*.h5'))
        print("[spDataset] found {} h5py files".format(len(h5py_files)))
        self.max_embed_dim = 0
        self.device = device
        self.lin_sep = lin_sep
        
        ##### TODO: func
        ep2cat_path = os.path.join(conf.dataset_path, 'ep2cat.json')
        with open(ep2cat_path) as json_file: 
            self.ep2cat = json.load(json_file)
        ep2cat_map_path = os.path.join(conf.dataset_path, 'ep2cat_mapping.json')
        with open(ep2cat_map_path) as json_file: 
            self.ep2cat_map = json.load(json_file)
        self.num_cats = len(self.ep2cat_map.keys())
        print("Number of cats: ", self.num_cats)
        ######

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

            for sent_idx in range(len(f['sentences'])):
                
                # Only load data for wich we have a category label
                if lin_sep:
                    episode = f['seg_ts'][sent_idx].decode("utf-8").split('_')[0]
                    if episode not in self.ep2cat.keys():
                        continue
                        
                        
                idx2file[sample_idx] = (h5idx, sent_idx)
                sample_idx += 1
                
                print("del del del!!")
                if sample_idx > 500:
                    break

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
            if self.load_full:
                h5py_idx, sent_idx = self.idx2file[index]
                h5py_file = self.h5py_idx2file[h5py_idx]

                f = h5py.File(h5py_file, 'r')
                sent = f['sentences'][sent_idx].decode("utf-8") 
                full_embeds = torch.Tensor(np.array(f[str(sent_idx)]))
                target = f['seg_ts'][sent_idx].decode("utf-8") 

                if self.lin_sep:
                    episode = target.split('_')[0]
                    cat = self.ep2cat[episode]
                    
                sample = (sent, full_embeds, target, cat)
            else:
                h5py_idx, sent_idx = self.idx2file[index]
                h5py_file = self.h5py_idx2file[h5py_idx]

                f = h5py.File(h5py_file, 'r')
                sent = f['sentences'][sent_idx].decode("utf-8") 
                target = f['seg_ts'][sent_idx].decode("utf-8") 
                mean_embeds = torch.Tensor(f['mean_embeddings'][sent_idx])
                
                if self.lin_sep:
                    episode = target.split('_')[0]
                    cat = self.ep2cat[episode]
                sample = (sent, mean_embeds, target, cat)

        elif self.load_full:
            h5py_idx, sent_idx = self.idx2file[index]
            h5py_file = self.h5py_idx2file[h5py_idx]

            f = h5py.File(h5py_file, 'r')
            sent = f['sentences'][sent_idx].decode("utf-8")
            full_embeds = torch.Tensor(np.array(f[str(sent_idx)]))
            sample = (sent, full_embeds)
            
        else:
            h5py_idx, sent_idx = self.idx2file[index]
            h5py_file = self.h5py_idx2file[h5py_idx]

            f = h5py.File(h5py_file, 'r') 
            sent = f['sentences'][sent_idx].decode("utf-8")
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
            targs = []
            for example in batch:
                targs.append(example[2])
            return text_embeds, padded_audio_embeds, lengths, targs, full_text, torch.tensor(cats)
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
            targs = []
            for example in batch:
                targs.append(example[2])
            return text_embeds, audio_embeds, lengths, targs, full_text, torch.tensor(cats)
        else:
            return text_embeds, audio_embeds, lengths

class spDatasetWeakShuffleLinSep(datautil.Dataset):
    """
    Spotify Podcast dataset dataloader. 
    Weak shuffling to enable shuffle while clustering sentences of similar length.
    """
    def __init__(self, CFG, directory=conf.sp_sample_path, traintest="train", load_full=False,lin_sep=False, device=None):
        print("[spDataset] init from directory [WEAKSHUFFLE] ", directory, traintest)
        directory = os.path.join(directory, traintest)
        h5py_files = list(Path(directory).glob('*.h5'))
        print("[spDataset] found {} h5py files".format(len(h5py_files)))
        self.max_embed_dim = 0
        self.device = device
        self.lin_sep = lin_sep
        
        ### DELETE
        CFG.max_train_samples = 128 * 10
        
        # TODO: func
        ep2cat_path = os.path.join(conf.dataset_path, 'ep2cat.json')
        with open(ep2cat_path) as json_file: 
            self.ep2cat = json.load(json_file)
        ep2cat_map_path = os.path.join(conf.dataset_path, 'ep2cat_mapping.json')
        with open(ep2cat_map_path) as json_file: 
            self.ep2cat_map = json.load(json_file)
        self.num_cats = len(self.ep2cat_map.keys())
        print("Number of cats: ", self.num_cats)

        idx2file = {}
        self.h5py_idx2file = h5py_files
        sample_idx = 0

        self.load_full = load_full
        self.traintest = traintest

        self.tokenizer = AutoTokenizer.from_pretrained(CFG.text_model_name)
        self.text_max_length = CFG.text_max_length
        self.file_startstop = []

        for h5idx, h5py_file in enumerate(h5py_files):    
            f = h5py.File(h5py_file, 'r')
            print("[spdataset] loading {}/{}: {}".format(h5idx, len(h5py_files), h5py_file))
            self.max_embed_dim = max(self.max_embed_dim, f.attrs['max_embed_dim'])  # TODO: can be removed?
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
            cats =[]

            last_idx = self.last_idx
            for enum, i in enumerate(index):
                h5py_idx, sent_idx = self.idx2file[i]
                if h5py_idx != last_idx:
                    # Clear memory
                    self.f.close()
                    # del self.mean_embeds
                    gc.collect()
                    print("[del3] Garbage collected")
                    h5py_file = self.h5py_idx2file[h5py_idx]
                    self.f = h5py.File(h5py_file, 'r')
                    print("[del3] loaded new h5py file: ", h5py_idx, h5py_file)
                
                # sent = f['sentences'][sent_idx]
                sent = self.f['sentences'][sent_idx].decode("utf-8")
                full_embeds = torch.Tensor(np.array(self.f[str(sent_idx)]))

                # Collate fn
                full_text.append(sent)
                text_embeds.append(sent)
                audio_embeds.append(full_embeds)
                lengths.append(len(full_embeds))
                
                if self.lin_sep:
                    episode = self.f['seg_ts'][sent_idx].decode("utf-8").split('_')[0]
                    cats.append(self.ep2cat[episode])

                last_idx = h5py_idx
            self.last_idx = last_idx

             # Pad the audio embeddings
            audio_embeds = pad_sequence(audio_embeds, batch_first=True).to(self.device)
            
            # Tokenize text
            text_embeds = self.tokenizer(
                text_embeds, padding=True, truncation=True, max_length=self.text_max_length, return_tensors='pt'
            ).to(self.device)


        else:
            text_embeds = []
            audio_embeds = []
            lengths  = []
            cats = []
            
            last_idx = self.last_idx

            for enum, i in enumerate(index):
                h5py_idx, sent_idx = self.idx2file[i]

                if h5py_idx != last_idx:
                    # Clear memory
                    self.f.close()
                    del self.mean_embeds
                    gc.collect()
                    print("Garbage collected")

                    h5py_file = self.h5py_idx2file[h5py_idx]
                    self.f = h5py.File(h5py_file, 'r')
                    print("[del2] loaded new h5py file: ", h5py_idx, h5py_file)
                    s1 = time()
                    self.mean_embeds = torch.Tensor(np.array(self.f['mean_embeddings']))
                    print("[del2]              it took: ", time() - s1)

                sent = self.f['sentences'][sent_idx].decode("utf-8")
                mean_embeds = self.mean_embeds[sent_idx]

                text_embeds.append(sent)
                audio_embeds.append(mean_embeds)
                
                if self.lin_sep:
                    episode = self.f['seg_ts'][sent_idx].decode("utf-8").split('_')[0]
                    cats.append(self.ep2cat[episode])
                    
                last_idx = h5py_idx

            self.last_idx = last_idx

            # Combine audio embeddings to a Tensor
            audio_embeds = torch.stack(audio_embeds).to(self.device)

            # Tokenize text
            max_length = 32 # TODO: is dit nodig?
            text_embeds = self.tokenizer(
                text_embeds, padding=True, truncation=True, max_length=max_length, return_tensors='pt'
            ).to(self.device)

        if self.lin_sep:
            return text_embeds, audio_embeds, lengths, torch.tensor(cats)
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
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        # self.batch_ids = torch.randperm(int(self.n_batches)) # sorted

        # Not shuffled:
        #self.batch_ids = torch.arange(int(self.n_batches))

        # Shuffled:
        self.batch_ids = self.shuffle_within_file(dataset)


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


####### UTILS
def freeze_network(model):
    for name, p in model.named_parameters():
        p.requires_grad = False

###### Linear Evaluation
class LinearEvaluationModel(nn.Module):
    """
        This loss expects as input a batch consisting of ... etc
    """
    def __init__(self, full_model, CFG, data_loader, modality='text'):
        """
        test
        """
        super(LinearEvaluationModel, self).__init__()
        self.text_model = full_model.text_model
        self.audio_model = full_model.audio_model

        self.device = torch.device(CFG.device)
        self.batch_size = full_model.batch_size
        
        freeze_network(self.text_model)
        freeze_network(self.audio_model)

        self.hidden_dim = CFG.final_projection_dim
        self.output_dim = data_loader.train_dataset.num_cats

        if not args.mlp:
            self.projectionhead = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim))
        else:
            self.projectionhead = nn.Sequential(
               nn.Linear(self.hidden_dim, self.hidden_dim),
               nn.ReLU(),
               nn.Linear(self.hidden_dim, self.output_dim),
            )
        self.criterion = nn.CrossEntropyLoss()
        self.modality = modality

        self.projectionhead.to(self.device)
        
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], audio_features: Tensor, seq_len, cats):

        with torch.no_grad():

            # Encode features
            reps_text = self.text_model(sentence_features)['sentence_embedding']
            reps_audio = self.audio_model((audio_features, seq_len))

            # Normalise features
            reps_audio = reps_audio / reps_audio.norm(dim=1, keepdim=True)
            reps_text = reps_text / reps_text.norm(dim=1, keepdim=True)
        
        if self.modality == 'text':
            preds = self.projectionhead(reps_text)
        else:
            preds = self.projectionhead(reps_audio)

        loss = self.criterion(preds, cats)
        metrics = self.get_metrics(preds.detach().cpu(), cats)
        return loss, metrics
        
    def get_metrics(self, preds, targets):
        metrics = {}
        y_pred = torch.argmax(preds, axis=1)
        
        print("\t\t\t y_pred: ", torch.unique(y_pred), torch.bincount(targets))
        
        metrics['acc'] = torch.sum(y_pred == targets) / targets.shape[0]
        metrics['targets'] = targets.tolist()
        metrics['preds'] = y_pred.tolist()
        
        return metrics

    
class LinearEvalator(nn.Module):
    """
        This loss expects as input a batch consisting of ... etc
    """
    def __init__(self, lin_eval_model, CFG, data_loader, modality='text'):
        """
        test
        """
        super(LinearEvalator, self).__init__()
        self.modality=modality
        self.lin_eval_model = lin_eval_model
        
        self.data_loader = data_loader
        self.output_path = CFG.log_name

        # to config file
        self.lin_max_epochs = 2
        self.in_batch_size = 256
        lin_lr = 0.001
        lin_weight_decay = 1.0e-6
        
        self.optimizer = torch.optim.Adam(
            self.lin_eval_model.parameters(),
            lr=lin_lr,
            weight_decay=lin_weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=False,
        )

        
    def fit(self):
        #self.lin_eval_model.projectionhead.train()
        self.acc_per_epoch = []
        for epoch in range(self.lin_max_epochs):
            print("-- -- -- Epoch: ", epoch)
            accs =[]
            for step, batch in enumerate(iter(self.data_loader.train_loader)):
                sent_features, audio_features, seq_len, cats = batch
                self.optimizer.zero_grad()

                loss, metrics = self.lin_eval_model(sent_features, audio_features, seq_len, cats)    
                accs.append(metrics['acc'])
                print("Loss: {} \t acc: {}".format(loss, metrics['acc']))

                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss)
                
                if step > 2:
                    print("delete this")
                    break

            print("-- Train epoch Mean acc: ", np.mean(accs))
            self.acc_per_epoch.append(np.mean(accs))
        print("Train done, accs per epoch: ", self.acc_per_epoch)

        
    def evaluate(self):
        accs = []
        self.lin_eval_model.projectionhead.eval()
        preds = []
        targets = []
        
        print("-- -- -- Start evaluation: ")
        with torch.no_grad():
            for step, batch in enumerate(iter(self.data_loader.test_loader)):
                sent_features, audio_features, seq_len, targs, _, cats = batch
                #self.optimizer.zero_grad()

                loss, metrics = self.lin_eval_model(sent_features, audio_features, seq_len, cats)    
                accs.append(metrics['acc'])
                print("eval Loss: {} \t acc: {}".format(loss, metrics['acc']))
                
                preds.extend(metrics['preds'])
                targets.extend(metrics['targets'])

        self.eval_mean_acc = np.mean(accs)
        self.preds = preds
        self.targets = targets
        
    def save_results(self):
        # Save results
        classes = list(self.data_loader.test_dataset.ep2cat_map.keys())
        self.make_cm(self.preds, self.targets, classes)
        
        print("type: ", self.acc_per_epoch)
        lin_eval_res = {'eval_acc': self.eval_mean_acc,
                       'acc_per_epoch': list(self.acc_per_epoch)}
        print("This is lin_eval_res: ", lin_eval_res)
        out = os.path.join(self.output_path, 
            '{}_linear_evaluation_results.csv'.format(self.modality))
        df = pd.DataFrame(lin_eval_res)
        df.T.to_csv(out, sep=';')
        print(df.T.to_string())

    def make_cm(self, y_pred, y_true, classes):
        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index = [i for i in classes],
            columns = [i for i in classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        
        out = os.path.join(self.output_path, 
            '{}_linear_evaluation_cm.png'.format(self.modality))
        print("Saving to: ", out)
        plt.savefig(out)
        # plt.show()


def main(args):
    model_path = args.model_path
    modality = args.modality
    print("[Load model] from ", model_path)
    model_weights_path = os.path.join(model_path, "output/full_model_weights.pth")
    model_config_path = os.path.join(model_path, 'config.json')

    # Opening JSON file
    f = open(model_config_path)
    model_config = json.load(f)
    fullcfg = from_dict(data_class=Cfg, data=model_config)
    fullcfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Load model] config loaded: ", fullcfg)

    # Create dataloader
    data_loader = MMloader(fullcfg, lin_sep=True)

    # Load the model
    full_model = mmModule(fullcfg)
    full_model.load_state_dict(torch.load(model_weights_path,  map_location=fullcfg.device))              
    full_model = full_model.to(fullcfg.device)     
    full_model.eval()

    # Perform evaluation
    lin_eval_model = LinearEvaluationModel(full_model, fullcfg, data_loader)
    evaluator = LinearEvalator(lin_eval_model, fullcfg, data_loader, modality=modality)
    evaluator.fit()
    evaluator.evaluate()
    evaluator.save_results()


if __name__ == "__main__":
    print("[evaluate] reading data from: ")
    print(conf.dataset_path)

    # Parse flags in command line arguments
    parser = ArgumentParser()

    parser.add_argument('--model_path', type=str, default="logs/AWS_run2-clip_loss_None_sph_768_False_2022-06-12_12-44-00",
                        help='Folder where model weights are saved.')
    parser.add_argument('--modality', default='text', const='text',
                    nargs='?', choices=['text', 'audio'],
                    help='Which modality to perform validation on (default: %(default)s)')
    
    # parser.add_argument('--save_intermediate', action='store_true', default=False,
    #                 help='Whether to save intermediate embeddings.')
    # parser.add_argument('--tmp_files_path', type=str, default="logs/windows_gru2_15m",
    #                     help='Folder where model weights are saved.')
    parser.add_argument('--mlp', action='store_true', default=False,
                    help='Whether to use multiple layers.')
    args, unparsed = parser.parse_known_args()

    main(args)
