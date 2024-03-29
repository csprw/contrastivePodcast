"""
Self-supervised Contrastive Learning from Podcast Audio and Transcripts.
Summary evaluation task
Author: Casper Wortmann
Usage: python evaluate_summary.py
"""
import sys
import os
from collections import defaultdict
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from nltk import tokenize
import gc
import h5py

import torch
from torch.utils import data as datautil
from torch.utils.data import Sampler, BatchSampler, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from dacite import from_dict
from argparse import ArgumentParser
from omegaconf import OmegaConf
from utils import randomize_model
from train import Cfg, mmModule
from dataloader import RandomBatchSampler

class MMloader_summary(object):
    """
    Module that loads the summary dataset. 
    Inputs:
        CFG: configuration of the current run. (See train.py).
        directory: path to the directory of the summary .h5py files. 
    """
    def __init__(self, CFG, directory):
        self.batch_size = CFG.batch_size
        self.directory = directory
        self.device = CFG.device
        self.load_full = True if CFG.audio_proj_head in ['gru', 'rnn', 'lstm', 'mlp'] else False

        # Create a dataloader that returns summaries and summary-episodes of the summary test-set. 
        self.episodes_dataset, self.episodes_loader = self.create_dataset_summary(CFG,  traintest="test_summary")

    def create_dataset_summary(self, CFG, traintest):
        """
        Create a dataloader for the summary test-set. 
        Inputs:
            CFG: Configuration instance. 
            traintest: The name of the test set (test_summary). 
        Outputs:
            dataset: the dataset of the summary set. 
            laoder: dataloader that returns samples of the summary set. 
        """
        dataset = self.get_sp_dataset(CFG, directory=self.directory,  traintest=traintest, load_full=self.load_full)
        loader = DataLoader(
            dataset, batch_size=None,  # must be disabled when using samplers
            sampler=BatchSampler(RandomBatchSampler(dataset, self.batch_size), 
            batch_size=self.batch_size, drop_last=True)
        )
        return dataset, loader

    def get_sp_dataset(self, CFG, directory, traintest="train", load_full=False):
        """ Generates the dataset itself. """
        dataset = spDatasetWeakShuffleLinSep(CFG, directory=directory, traintest=traintest,  load_full=load_full, device=self.device)
        return dataset

class spDatasetWeakShuffleLinSep(datautil.Dataset):
    """
    Spotify Podcast dataset dataloader. 
    Weak shuffling to enable shuffle while clustering sentences of similar length.
    """
    def __init__(self, CFG, directory, traintest="train", load_full=False, device=None):
        """
        Inputs:
            CFG: Configuration settings of the current run.
            Directory: path to the SP dataset.
            traintest: The test split to use (test_summary). 
            load_full: if set to True return all yamnet embeddings directly,
                else average yamnets before returning them. 
            device: cpu or gpu specification. 
        """
        
        directory = os.path.join(directory, traintest)
        h5py_files = list(Path(directory).glob('*.h5'))
        print("[spDataset] found {} h5py files".format(len(h5py_files)))

        self.device = device
        self.return_targets = True
        
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
                tmp = f['seg_ts'][sent_idx].decode("utf-8")
                episode = tmp.split('_')[0]
                idx2file[sample_idx] = (h5idx, sent_idx, episode)
                sample_idx += 1
        
            f.close()
            self.file_startstop.append((start_idx, sample_idx))
        
        self.idx2file = idx2file
        self.last_idx = -1
        self.mean_embeds = None
        self.f =  h5py.File(h5py_file, 'r')

    def __len__(self):
        """ Denotes the total number of utterances. """
        return len(self.idx2file.keys())

    def __getitem__(self, index):
        """ Return one item from the df. """
        text_embeds = []
        audio_embeds = []
        full_text = []
        lengths  = []
        cats = []
        targs = []
        sentences = []

        last_idx = self.last_idx
        for enum, i in enumerate(index):
            h5py_idx, sent_idx, episode = self.idx2file[i]
            if h5py_idx != last_idx:
                # Clear memory
                self.f.close()
                if not self.load_full:
                    del self.mean_embeds
                gc.collect()
                h5py_file = self.h5py_idx2file[h5py_idx]
                self.f = h5py.File(h5py_file, 'r')
                if not self.load_full:
                    self.mean_embeds = torch.Tensor(np.array(self.f['mean_embeddings']))
            
            sent = self.f['sentences'][sent_idx].decode("utf-8")
            sentences.append(sent)
            if self.load_full:
                audio_embed = torch.Tensor(np.array(self.f[str(sent_idx)]))
                lengths.append(len(audio_embed))
            else:
                audio_embed = self.mean_embeds[sent_idx]

            full_text.append(sent)
            text_embeds.append(sent)
            audio_embeds.append(audio_embed)

            episode = self.f['seg_ts'][sent_idx].decode("utf-8").split('_')[0]
            targs.append(episode)
            last_idx = h5py_idx
        self.last_idx = last_idx

        # Pad the audio embeddings
        audio_embeds = pad_sequence(audio_embeds, batch_first=True).to(self.device)
        
        # Tokenize text
        text_embeds = self.tokenizer(
            text_embeds, padding=True, truncation=True, max_length=self.text_max_length, return_tensors='pt'
        ).to(self.device)

        return text_embeds, audio_embeds, lengths, targs, sentences, cats



class summaryEvaluator(object):
    def __init__(self, CFG, model_path, full_model, data_loader):
        """
        Evaluator object to perform a summary ranking task. 
        Inputs:
            CFG: configuration of current run. 
            model_path: path to the run we want to evaluate. 
            full_model: the model with the loaded weights. 
            data_loader: a summary dataoader. 
        """
        self.model_path = model_path
        self.model = full_model
        self.device = CFG.device
        self.scale = CFG.scale
        self.bs = CFG.batch_size
        self.embed_dim = CFG.mutual_embedding_dim
        self.tokenizer =  data_loader.episodes_dataset.tokenizer
        
        self.episodes_dataset = data_loader.episodes_dataset
        self.episodes_loader = data_loader.episodes_loader
        self.summaries_loader = None

        self.audio_proj_head = CFG.audio_proj_head
        self.fixed_scale = CFG.scale
        
        self.test_dir = os.path.join(CFG.sp_path, 'test_summary')
    
    def get_max_episode_data(self):
        """
        Returns max number of sentences to encode.
        """
        max_samples =  len(self.episodes_dataset)
        return max_samples

    def audio_to_embed(self, yamnets, query_lengths):
        """
        Creates summary embeddings from yamnet embeddings.
        Inputs:
            yamnets: a list of yamnet embeddings. 
            query_lengths: the lengths of the yamnet embeddings before padding. 
        Output:
            embed: an embedding of the audio squence given the current model. 
        """
        if self.audio_proj_head in ['gru', 'rnn', 'lstm', 'mlp']:
            padded_yamnets = pad_sequence(yamnets, batch_first=True).to(self.device)

            with torch.no_grad():
                reps_audio = self.model.audio_model((padded_yamnets, query_lengths))
                embed = reps_audio / reps_audio.norm(dim=1, keepdim=True)
        else:
            with torch.no_grad():
                yamnets_mean = [torch.tensor(y.mean(dim=0).clone().detach()) for y in yamnets]

                audio_features = torch.stack(yamnets_mean).to(self.device)
                reps_audio = self.model.audio_model((audio_features, query_lengths))
                embed = reps_audio / reps_audio.norm(dim=1, keepdim=True)
        return embed.cpu()

    def text_to_embed(self, text):
        """
        Creates summary embeddings from symmary texts.
        Inputs:
            text: a list of strings (sentences). 
        Outputs:
            embed: the text embddings, output of the text encoder. 
        """
        tokenized_text = self.tokenizer(
            text, padding=True, truncation=True, max_length=32, return_tensors='pt', return_token_type_ids=True,
        )
        
        with torch.no_grad():
            tokenized_text = tokenized_text.to(self.device)
            reps_sentences = self.model.text_model(tokenized_text)['sentence_embedding']
            embed = reps_sentences / reps_sentences.norm(dim=1, keepdim=True)

        return embed.cpu()

    @torch.no_grad()  
    def encode_summary_episodes(self, max_samples):
        """
        Creates representations of sentences in all episodes in the summary test-set. 
        Inputs:
            max_samples: the number of samples to encode.
        """
        text_encoding = np.zeros((max_samples, self.embed_dim)) 
        audio_encoding = np.zeros((max_samples, self.embed_dim)) 
        
        all_sents = np.zeros(max_samples, dtype=object)
        all_targs = np.zeros(max_samples, dtype=object)
        
        for step, batch in enumerate(self.episodes_loader):
            (text_embeds, audio_embeds, seq_len, targets, sents, _) = batch
            print("Calculating: ", step*self.bs, max_samples)
            text_embeds = text_embeds.to(self.device)
            audio_embeds = audio_embeds.to(self.device)  

            reps_sentences = self.model.text_model(text_embeds)['sentence_embedding']
            reps_audio = self.model.audio_model((audio_embeds, seq_len))

            audio_batch = reps_audio / reps_audio.norm(dim=1, keepdim=True)
            text_batch = reps_sentences / reps_sentences.norm(dim=1, keepdim=True)

            cur_idx = step * self.bs
            next_idx = cur_idx + len(sents)
            text_encoding[cur_idx:next_idx] = text_batch.cpu().numpy()
            audio_encoding[cur_idx:next_idx] = audio_batch.cpu().numpy()
            
            all_sents[cur_idx:next_idx] = sents
            all_targs[cur_idx:next_idx] = targets

            if next_idx >= max_samples: 
                print("Max samples reached")
                break
          
        self.all_sents = all_sents
        self.all_targs = all_targs
        self.text_encoding = text_encoding
        self.audio_encoding = audio_encoding

    @torch.no_grad()  
    def encode_summaries(self, summary_dict, summary_audio_dict):
        """
        Creates representations of all summaries in the summary test-set. 
        Pushes the entire summary at once through the encoder. 
        Inputs:
            summary_dict: a dict consisting of episodes and their summaries (text).
            summary_audio_dict: a dict consisting of episodes and their summaries (yamnet embeddings). 
        """
        summary_targets = []
        summary_text_encodings = []
        summary_texts = []
        summary_audio_encodings = []

        for epi, summlist in summary_dict.items():
            audio_embeds = summary_audio_dict[epi]
            
            for summ_text, yamnet_embed in zip(summlist, audio_embeds):
                summary_targets.append(epi)
                summ_text = tokenize.sent_tokenize(summ_text)[0]  
                summary_text_encoding = self.text_to_embed(summ_text)[0]

                summary_text_encodings.append(summary_text_encoding)
                summary_texts.append(summ_text)
                query_length = len(yamnet_embed)
                summary_audio_encoding = self.audio_to_embed([yamnet_embed], [query_length]).cpu()
                summary_audio_encodings.append(summary_audio_encoding)

        self.summary_targets = summary_targets
        self.summary_text_encoding = torch.stack(summary_text_encodings)
        self.summary_texts = summary_texts
        self.summary_audio_encoding = torch.vstack(summary_audio_encodings)
        
    @torch.no_grad()  
    def encode_summaries_sentlevel(self, sent_summary_dict, sent_summary_audio_dict):
        """
        Creates representations of sentences in all summaries in the summary test-set. 
        Pushes the each sentence of the summary seperately through the encoder. 
        Inputs:
            summary_dict: a dict consisting of episodes and their summaries (text).
            summary_audio_dict: a dict consisting of episodes and their summaries (yamnet embeddings). 
        """
        sent_summ_targets = []
        sent_summ_text_encodings = []
        sent_summ_texts = []
        sent_summ_audio_encodings = []
        
        for epi, summ_text in sent_summary_dict.items():
            audio_embeds = sent_summary_audio_dict[epi]
            sent_summ_targets.append(epi)
                
            sent_summ_text = tokenize.sent_tokenize(summ_text)  
            sent_summ_text_encoding = self.text_to_embed(sent_summ_text)[0]
            sent_summ_text_encodings.append(sent_summ_text_encoding)
            sent_summ_texts.append(summ_text)

            query_length = len(audio_embeds[0])
            summary_audio_encoding = self.audio_to_embed([audio_embeds[0]], [query_length]).cpu()
            sent_summ_audio_encodings.append(summary_audio_encoding)
 
        self.sent_summ_targets = sent_summ_targets
        self.sent_summ_text_encoding = torch.stack(sent_summ_text_encodings)
        self.sent_summ_texts = sent_summ_texts
        self.sent_summ_audio_encoding = torch.vstack(sent_summ_audio_encodings)

def get_summ_audio(summary_embed_dir):
    """
    Returns audio fragments of summaries in a directory.
    Inputs:
        summary_embed_dir: path to a folder containing summary yamnet embeddings.
    Output:
        summary_audio_dict: dict containing episodes and the audio fragments of the summaries. 
    """
    # Read the audio embeddings of the summaries
    summary_audio_dict = defaultdict(list)

    pathlist = Path(summary_embed_dir).glob('**/*.h5')
    for path in pathlist:
        filename = str(path)

        with h5py.File(filename, "r") as f:
            
            filename = os.path.split(filename)[-1]
            if '_' in filename:
                target_name = filename.split('_')[0]
            else:
                target_name = filename.split('.')[0]

            emb = torch.Tensor(np.array(f['embedding']['block0_values']))
            summary_audio_dict[target_name].append(emb)

    return summary_audio_dict

def get_summ_sent_audio(sent_summary_embed_dir):
    """
    Returns audio fragments of summaries in a directory.
    Inputs:
        sent_summary_embed_dir: path to a folder containing summary yamnet embeddings (per sentence).
    Output:
        sent_summary_audio_dict: dict containing episodes and the audio fragments of the summaries (per sentence)
    """
    # Read summary embeddings into a dict
    sent_summary_audio_dict = defaultdict(list)

    pathlist = Path(sent_summary_embed_dir).glob('**/*.h5')
    for path in pathlist:
        # because path is object not string
        filename = str(path)

        with h5py.File(filename, "r") as f:
            filename = os.path.split(filename)[-1]
            target_name = filename.split('.')[0]
            emb = torch.Tensor(np.array(f['embedding']['block0_values']))
            sent_summary_audio_dict[target_name].append(emb)

    return sent_summary_audio_dict


def summary_evaluation(evaluator, target='sent'):
    """
    Creates similarity matrices between the summary encodings and 
    episode encodings. 
    Inputs:
        evaluator: an instance of Evaluator class.
        target: Wheter to push the entire summary at once
            through the encoder ('sent'), or each sentence seperately ('full')
    """
    assert target in [ 'sent', 'full']
    if target == 'sent':
        # Results on sent-summary level, each sentence was pushed seperately
        # through the encoder
        summary_encodings = [(evaluator.sent_summ_text_encoding, 'sentsummtext'),
                        (evaluator.sent_summ_audio_encoding, 'sentsummaudio')]
        targets = evaluator.sent_summ_targets
    else:
        # Sentences are not split, the summary was pushed at once through encoder.
        summary_encodings = [(evaluator.summary_text_encoding, 'fullsummtext'),
                        (evaluator.summary_audio_encoding, 'fullsummaudio')]
        targets = evaluator.summary_targets
    epi_encodings = [(evaluator.text_encoding, 'text'),
                    (evaluator.audio_encoding, 'audio')]

    k = evaluator.text_encoding.shape[0]
    results = defaultdict(list)
    for summary_tup in summary_encodings:
        for tup in epi_encodings:
            name = summary_tup[1] + "-" + tup[1]
            print("------- Results for: ", name)
            summ_encoding = summary_tup[0]
            epi_encoding = tup[0]
            
            # Create similarity matrix.
            similarity = (100.0 * summ_encoding @ epi_encoding.T).softmax(dim=-1)
            rank = []        
            mrr = []
            confidence = []
            total_indices = []
            idxs= []
            for idx in range(len(targets)):
                # Calculate topK.
                values, indices = similarity[idx].topk(k)
                target = targets[idx].split("_")[0]
                
                confidence.extend(values)
                total_indices.extend(indices)
                idxs.append(idx)
                    
                # Continue untill we have the ranks of all sentences in the summary.
                if idx != (len(targets) - 1):
                    next_target = targets[idx+1].split("_")[0]
                    if next_target == target:
                        continue
                    
                # Combine the rankings of all sentences in the summary to one rank. 
                confidence = np.array(confidence)
                total_indices = np.array(total_indices)
                sorted_inds = confidence.argsort()[::-1]
                total_indices = total_indices[sorted_inds]
                predicted_epis = evaluator.all_targs[total_indices.tolist()].tolist()

                # Calculate best rank.
                if target in  predicted_epis:
                    estimated_position = predicted_epis.index(target)
                    rank.append( estimated_position)
                    mrr.append(1 / (estimated_position + 1))

                confidence = []
                total_indices = []
                idxs= []

            print("Mean position: {} (std: {}) \t mrr: {} ".format(np.mean(rank), np.std(rank), np.mean(mrr)))
            results['names'].append(name)
            results['ranks'].append(np.mean(rank))
            results['ranks_sd'].append(np.var(rank))
            results['mrrs'].append(np.mean(mrr))
            results['mrrs_sd'].append(np.var(mrr))
    return results

def to_plot(results, target='sent'):
    """
    Saves the results to plots. 
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)
    offset = 0.0
    width = 0.5

    ax.set_title('Ranking Query summaries', fontsize=16)
    ax.yaxis.grid(False)
    ax.set_ylabel("MRR", fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    names = results['names']
    name_vals =np.arange(len(names))
    label = os.path.normpath(args.model_weights_path).split(os.path.sep)[1]
    output_path = os.path.split(args.model_weights_path)[0]
    ax.bar(x=name_vals + (1*offset), height=results['mrrs'], yerr=results['mrrs_sd'], width=width, ecolor='black', color='darkred', alpha=0.8,  label=label)

    ax.set_xticks(name_vals, names)
    plt.xticks(rotation=30)

    # Save the figure. 
    plt.legend(loc='best',  bbox_to_anchor=(1.0,1.0))
    plt.tight_layout()
    if target != 'sent':
        plt.savefig(os.path.join(output_path, 'summary_full.png'))
        plt.savefig(os.path.join(output_path, 'summary_full.pdf'))
    else:
        plt.savefig(os.path.join(output_path, 'summary_sent.png'))
        plt.savefig(os.path.join(output_path, 'summary_sent.pdf'))


def main(args):
    """
    Loads the model, the test set, and targets. 
    Inputs:
        args: argpars instance. 
    """
    conf = OmegaConf.load("./config.yaml")
    summaries_output_path = os.path.join(conf.dataset_path, 'TREC', 'good_summaries.json')
    sent_summaries_output_path = os.path.join(conf.dataset_path, 'TREC', 'sent_summaries.json')

    ## Read dictionary with summaries
    with open(summaries_output_path, 'r') as f:
        summary_dict = json.load(f)
    with open(sent_summaries_output_path, 'r') as f:
        sent_summary_dict = json.load(f)

    model_weights_path = args.model_weights_path
    model_path = str(Path(model_weights_path).parents[1])
    model_config_path = os.path.join(model_path, 'config.json')

    # Opening JSON file
    f = open(model_config_path)
    model_config = json.load(f)
    fullcfg = from_dict(data_class=Cfg, data=model_config)
    fullcfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    fullcfg.sp_path = conf.sp_path

    # Create dataloader
    data_loader = MMloader_summary(fullcfg, directory=conf.yamnet_processed_path)

    # Load the model
    full_model = mmModule(fullcfg)
    full_model.load_state_dict(torch.load(model_weights_path,  map_location=fullcfg.device))    
    if args.create_random:
        randomize_model(full_model)          
    full_model = full_model.to(fullcfg.device)     
    full_model.eval()

    # Get yamnet embeddings
    summary_audio_dict = get_summ_audio(conf.summary_embed_dir)
    sent_summary_audio_dict = get_summ_sent_audio(conf.sent_summary_embed_dir)

    # Create evaluator en encode episodes and summaries
    evaluator = summaryEvaluator(fullcfg, model_path, full_model, data_loader)
    max_episodes = evaluator.get_max_episode_data()
    evaluator.encode_summary_episodes(max_episodes)
    evaluator.encode_summaries(summary_dict, summary_audio_dict)
    evaluator.encode_summaries_sentlevel(sent_summary_dict, sent_summary_audio_dict)

    # Perform evaluation on sentence level
    results = summary_evaluation(evaluator, target='sent') 

    # Save the results
    to_plot(results, target='sent')
    json_out = str(Path(args.model_weights_path).parents[1])
    with open(os.path.join(json_out, 'summary_results_sent.json'), 'w') as fp:
        json.dump(results, fp, indent=4)

    # Perform evaluation by pushing the full summary at once through the encoder
    # (Only first 512 tokens are taken into account)
    results = summary_evaluation(evaluator, target='full') 

    # Save the results
    to_plot(results, target='full')
    json_out = str(Path(args.model_weights_path).parents[1])
    with open(os.path.join(json_out, 'summary_results_full.json'), 'w') as fp:
        json.dump(results, fp, indent=4)


if __name__ == "__main__":
    # Parse flags in command line arguments
    parser = ArgumentParser()

    parser.add_argument('--model_weights_path', type=str, default="./logs/30m-mlp_2022-07-05_10-54-42/output/full_model_weights.pt",
                        help='Folder where model weights are saved.')
    parser.add_argument('--create_random', action='store_true', default=False,
                    help='If set to true, the model is initialized using random weights.')
    args, unparsed = parser.parse_known_args()

    main(args)
