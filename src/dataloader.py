"""
Self-supervised Contrastive Learning from Podcast Audio and Transcripts.
Author: Casper Wortmann
"""
import os
from pathlib import Path
import numpy as np
import h5py
import torch
import random
import gc
from torch import nn, Tensor
from torch.utils import data as datautil
from torch.utils.data import Sampler, BatchSampler, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

class MMloader(object):
    """
    Module that load datasets. 
    """
    def __init__(self, CFG):

        train_dataset_name = CFG.train_dataset
        val_dataset_name = CFG.val_dataset
        test_dataset_name = CFG.test_dataset

        batch_size = CFG.batch_size
        self.batch_size = batch_size
        self.device = CFG.device

        self.load_full = True if CFG.audio_proj_head in ['gru', 'rnn', 'lstm', 'mlp'] else False

        # Get the datasets
        if train_dataset_name == "sp_sample":
            self.train_dataset = self.get_sp_dataset(CFG, directory=CFG.sp_sample_path, traintest="train", load_full=self.load_full, device=self.device)
        elif train_dataset_name == "sp":
            self.train_dataset = self.get_sp_dataset(CFG, directory=CFG.sp_path,  traintest="train", load_full=self.load_full, device=self.device)
        else:
            raise Exception('Unknown dataset')
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=None,  # must be disabled when using samplers
            sampler=BatchSampler(RandomBatchSampler(self.train_dataset, batch_size), batch_size=batch_size, drop_last=True)
        )
        print("[MMloader] train dataset loaded, length: ", len(self.train_dataset))

        if val_dataset_name == "sp_sample":
            self.val_dataset = self.get_sp_dataset(CFG, directory=CFG.sp_sample_path,  traintest="val", load_full=self.load_full, device=self.device)
        elif val_dataset_name == "sp":
            self.val_dataset  = self.get_sp_dataset(CFG, directory=CFG.sp_path,  traintest="val",  load_full=self.load_full, device=self.device)
        else:
            raise Exception('Unknown dataset')
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=None,  # must be disabled when using samplers
            sampler=BatchSampler(RandomBatchSampler(self.val_dataset , batch_size), batch_size=batch_size, drop_last=True)
        )
        print("[MMloader] val dataset loaded, length: ", len(self.val_dataset))

        if test_dataset_name == "sp_sample":
            self.test_dataset = self.get_sp_dataset(CFG, directory=CFG.sp_sample_path,  traintest="test", load_full=self.load_full, device=self.device)
        elif test_dataset_name == "sp":
            self.test_dataset = self.get_sp_dataset(CFG, directory=CFG.sp_path,  traintest="test",  load_full=self.load_full, device=self.device)
        else:
            raise Exception('Unknown dataset')
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=None,  # must be disabled when using samplers
            sampler=BatchSampler(RandomBatchSampler(self.test_dataset, batch_size, shuffle=False), batch_size=batch_size, drop_last=True)
        )
        print("[MMloader] test dataset loaded, length: ", len(self.test_dataset))

    def get_sp_dataset(self, CFG, directory, traintest="train", load_full=False, device=None):
        dataset = spDatasetWeakShuffle(CFG, directory=directory, traintest=traintest,  load_full=load_full, device=device)
        return dataset

class spDatasetWeakShuffle(datautil.Dataset):
    """
    Spotify Podcast dataset dataloader. 
    Weak shuffling to enable shuffle while clustering sentences of similar length.
    """
    def __init__(self, CFG, directory, traintest="train", load_full=False, device=None):
        print("[spDataset] init from directory ", directory, traintest)
        directory = os.path.join(directory, traintest)
        h5py_files = list(Path(directory).glob('*.h5'))
        print("[spDataset] found {} h5py files".format(len(h5py_files)))
        self.device = device

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
            start_idx = sample_idx

            for sentidx in range(len(f['sentences'])):
                idx2file[sample_idx] = (h5idx, sentidx)
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
        """ Denotes the total number of samples """
        return len(self.idx2file.keys())

    def __getitem__(self, index):
        """ Return one sample """
        if self.traintest == 'test':
            # TODO: check if this works after cleaning for self==load_full
            text_embeds = []
            audio_embeds = []
            full_text = []
            lengths  = []
            targets = []
            last_idx = self.last_idx

            for enum, i in enumerate(index):
                h5py_idx, sent_idx = self.idx2file[i]

                if h5py_idx != last_idx:
                    # Clear memory
                    self.f.close()
                    del self.mean_embeds
                    gc.collect()

                    if self.load_full:
                        h5py_file = self.h5py_idx2file[h5py_idx]
                        self.f = h5py.File(h5py_file, 'r')
                    else:
                        h5py_file = self.h5py_idx2file[h5py_idx]
                        self.f = h5py.File(h5py_file, 'r')
                        self.mean_embeds = torch.Tensor(np.array(self.f['mean_embeddings']))

                if self.load_full:
                    embeds = torch.Tensor(np.array(self.f[str(sent_idx)]))
                else:
                    embeds = self.mean_embeds[sent_idx]
                sent = self.f['sentences'][sent_idx].decode("utf-8")
                target = self.f['seg_ts'][sent_idx].decode("utf-8") 

                full_text.append(sent)
                text_embeds.append(sent)
                audio_embeds.append(embeds)
                lengths.append(len(embeds))
                targets.append(target)
                last_idx = h5py_idx
            self.last_idx = last_idx

            if self.load_full:
                # Pad the audio embeddings
                audio_embeds = pad_sequence(audio_embeds, batch_first=True).to(self.device)
            else: 
                audio_embeds = torch.stack(audio_embeds).to(self.device)

            # Tokenize text
            text_embeds = self.tokenizer(
                text_embeds, padding=True, truncation=True, max_length=self.text_max_length, return_tensors='pt'
            ).to(self.device)

            sample = (text_embeds, audio_embeds, lengths, targets, full_text)
            return sample

        else:
            text_embeds = []
            audio_embeds = []
            full_text = []
            lengths  = []
            last_idx = self.last_idx

            for enum, i in enumerate(index):
                h5py_idx, sent_idx = self.idx2file[i]
                if h5py_idx != last_idx:
                    # Clear memory
                    self.f.close()
                    gc.collect()

                    if self.load_full:
                        h5py_file = self.h5py_idx2file[h5py_idx]
                        self.f = h5py.File(h5py_file, 'r')
                    else:
                        h5py_file = self.h5py_idx2file[h5py_idx]
                        self.f = h5py.File(h5py_file, 'r')
                        self.mean_embeds = torch.Tensor(np.array(self.f['mean_embeddings']))


                sent = self.f['sentences'][sent_idx].decode("utf-8")

                if self.load_full:
                    embeds = torch.Tensor(np.array(self.f[str(sent_idx)]))
                else:
                    embeds = self.mean_embeds[sent_idx]

                # Collate fn
                full_text.append(sent)
                text_embeds.append(sent)
                audio_embeds.append(embeds)
                lengths.append(len(embeds))
                last_idx = h5py_idx

            self.last_idx = last_idx

            if self.load_full:
                # Pad the audio embeddings
                audio_embeds = pad_sequence(audio_embeds, batch_first=True).to(self.device)
            else: 
                audio_embeds = torch.stack(audio_embeds).to(self.device)

            # Tokenize text
            text_embeds = self.tokenizer(
                text_embeds, padding=True, truncation=True, 
                max_length=self.text_max_length, return_tensors='pt'
            ).to(self.device)

            return text_embeds, padded_audio_embeds, lengths

class RandomBatchSampler(Sampler):
    """
    Sampling class to create random sequential batches from a given dataset
    :returns: generator object of shuffled batch indices
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size

        if shuffle:
            self.batch_ids = self.shuffle_within_file(dataset)
        else:
            self.batch_ids = torch.arange(int(self.n_batches))

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
            for index in idx:
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(int(self.n_batches) * self.batch_size, self.dataset_length)
            for index in idx:
                yield int(index)
