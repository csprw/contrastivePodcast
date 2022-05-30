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

from transformers import DistilBertTokenizer
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

import src.data
from src.data import load_metadata, find_paths, relative_file_path

import sys
sys.path.append('../contrastive_mm2/') 
from gru_test2 import mmModule, Cfg

# Load static configuration variables. 
config_path = "./config.yaml"
conf = OmegaConf.load(config_path)
print("[cudacheck] Is cuda available? ", torch.cuda.is_available())

# @dataclass
# class Cfg:
#     batch_size: int = 128
#     num_epochs: int = 1
#     loss_type: str = 'simcse_loss'
#     lr: float = 5e-5
#     # device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"

#     # For the audio module
#     audio_encoder_input: int = 1024
#     audio_hidden_dim: int = 768
#     audio_layer_dim: int = 2
#     audio_activation: str = 'relu'

#     # For the text module
#     text_model_name : str = 'distilbert-base-uncased'
#     text_tokenizer: str = "distilbert-base-uncased"
#     text_pooling: str = 'mean'   #['cls', 'mean']
#     text_max_length: int = 32

#     # For the projection_head modules
#     text_proj_head: str = 'None'
#     audio_proj_head: str = 'None'
#     text_activation: str = 'gelu'
#     mutual_embedding_dim: int = 768


#     final_projection_dim: int = 768  # [256 or 768]
#     audio_dropout: float = 0.1
#     text_dropout: float = 0.1
#     weight_decay: float = 0.01

#     text_activation: str = 'Test'
#     audio_activation: str = 'Test'
#     scale_type: str = 'Test'
#     scale: int = 20
#     pad_pack: bool = False
#     normalize: bool = False
#     eval_every: int = 1
#     print_every: int = 1

#     train_dataset: str = 'Test'
#     test_dataset: str = 'test'
#     seed: int = 100

#     save_model: bool = False
#     save_checkpoint: bool = False
#     load_model_path: str = ''
#     load_model: bool = False
#     load_checkpoint: bool = False
#     load_checkpoint_path: str = ''


class Indexer:
    """Class to run search queries with."""

    def __init__(self, model, tokenizer, CFG, config_path):
        """Init method for the Searcher."""
        super().__init__()
        # Load the configuration
        conf = OmegaConf.load(config_path)
        print('[Indexer] init')
        
        self.tokenizer = tokenizer
        self.model = model 
        self.metadata = load_metadata(conf.dataset_path)
        
        # Load yamnet embeddings and scores
        self.yamnet_embed_dir = conf.yamnet_embed_dir

        self.device = CFG.device
        # self.layer_dim = CFG.layer_dim
        self.audio_hidden_dim = CFG.audio_hidden_dim

        self.model.to(self.device)
        self.audio_model_type = self.model.audio_model._modules['0'].__class__.__name__
        
    def tokenize_text(self, sentences):
        # Tokenize text
        tokenized_text = self.tokenizer(
            sentences, padding=True, truncation=True, max_length=32, return_tensors='pt', return_token_type_ids=True,
        )
        return tokenized_text

    def get_embeds(self, tokenized_text):
        # input_ids = tokenized_text['input_ids'].to(self.device)
        # attention_mask = tokenized_text['attention_mask'].to(self.device)

        with torch.no_grad():
            # print("[del] get embeds: ")
            embeds = self.model.text_model(tokenized_text)['sentence_embedding']
            # embeds = self.model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeds = embeds.detach().cpu().numpy()
            # print("[del] embeds: ", embeds)
            # print("[del] so good ;-{}}  } ")
            # exit(1)

            text_logits = embeds / embeds.norm(dim=1, keepdim=True)
        return text_logits

    def get_yamnets_from_timestamps(self, start_time, end_time, segment_id):
        ep_prefix, _ = segment_id.split("_")
        show_prefix = self.metadata.loc[self.metadata['episode_filename_prefix'] == ep_prefix]['show_filename_prefix'].item()
        filepath = relative_file_path(show_prefix, ep_prefix)

        embed_path = os.path.join(self.yamnet_embed_dir, filepath + ".h5")

        yamnet_embedding = pd.read_hdf(embed_path)
        inbetween_yamnet_embeds = yamnet_embedding[yamnet_embedding.index.to_series().between(start_time, end_time)]
        return inbetween_yamnet_embeds


    def get_yamnets_from_segid(self, segment_id):
        ep_prefix, timestamp = segment_id.split("_")
        show_prefix = self.metadata.loc[self.metadata['episode_filename_prefix'] == ep_prefix]['show_filename_prefix'].item()
        filepath = relative_file_path(show_prefix, ep_prefix)

        embed_path = os.path.join(self.yamnet_embed_dir, filepath + ".h5")
        start_time = float(timestamp)
        end_time = float(timestamp) + 120
        yamnet_embedding = pd.read_hdf(embed_path)
        inbetween_yamnet_embeds = yamnet_embedding[yamnet_embedding.index.to_series().between(start_time, end_time)]
        return inbetween_yamnet_embeds
    
    
        
    def process_yamnet_embeds(self, inbetween_yamnet_embeds):
        # if GRU or RNN:
        if self.audio_model_type == 'SequentialAudioModel':
            # print("The inbetweens: ", inbetween_yamnet_embeds)
            lengths = len(inbetween_yamnet_embeds)

            audio_embeds = torch.Tensor(np.array(inbetween_yamnet_embeds))
            # print("shape: ", audio_embeds.shape)

            audio_embeds = torch.unsqueeze(audio_embeds, 0)
            # print("shape batch first: ", audio_embeds.shape)
            seq_len = [lengths] # The module expects a list of lengths
            # print(" should be 251: ", lengths)
            padded_audio_embeds = pad_sequence(audio_embeds, batch_first=True).to(self.device)
            reps_audio = self.model.audio_model((padded_audio_embeds, seq_len))
            
            # Normalize embeddings
            audio_logits = reps_audio / reps_audio.norm(dim=1, keepdim=True)
            return audio_logits

        else:
            raise NotImplementedError

        
        # raise
        # if type(self.model.audio_encoder).__name__ == 'simple_ProjectionHead':
        #     audio_type = 'mean'
        #     lengths = None
        #     mean_embeds = torch.tensor(inbetween_yamnet_embeds.mean()).to(torch.float32)

        #     mean_embeds = mean_embeds.reshape(1,-1)
            
        #     self.model.eval()
        #     with torch.no_grad():
        #         embeds = self.model.audio_encoder(mean_embeds, lengths)
        #     return embeds[0], audio_type
        # elif type(self.model.audio_encoder).__name__ == 'SequentialAudioModel':
        #     #print("TODO: GRU GET_YAMNET_EMBEDS")
        #     audio_type = 'gru'
        #     x = torch.tensor(inbetween_yamnet_embeds.values).to(torch.float32)
        #     x = x.to(self.device)
        #     h0 = torch.zeros(self.layer_dim, 1, self.audio_hidden_dim).to(self.device)
            
        #     self.model.eval()
        #     with torch.no_grad():
        #         x = x[None,:,:]
        #         # embedded = torch.nn.utils.rnn.pack_padded_sequence(audio_embed, length, batch_first=True, enforce_sorted=False)
        #         out, h0 = self.model.audio_encoder.seq_model(x, h0.detach())
                
        #         test = h0[-1, :, :]
        #         out = self.model.audio_encoder.fc(test)
                
        #     #embeds = self.model.audio_encoder(inbetween_yamnet_embeds)
        #     return out[0].detach().cpu(), audio_type

    
    def get_yamnet_embeds(self, segment_id):
        inbetween_yamnet_embeds = self.get_yamnets_from_segid(segment_id)
        
        embeds, audio_type = self.process_yamnet_embeds(inbetween_yamnet_embeds)
        return embeds, audio_type


class Searcher:
    """Class to run search queries with."""

    def __init__(self, config_path="./config.yaml"):
        """Init method for the Searcher."""
        super().__init__()
        # Load the configuration
        conf = OmegaConf.load(config_path)
        self.es_url = conf.search_es_url  # URL of Elasticsearch to query
        self.es_num = (conf.search_es_num)  # Number of segments to request from Elasticsearch
        self.sample_rate = 44100  # Hardcoded sample rate of all podcast audio

        self.dataset_path = conf.dataset_path
        self.audio_path = os.path.join(conf.dataset_path, "podcasts-audio")
        self.metadata = load_metadata(conf.dataset_path)
        
        # Load yamnet embeddings and scores
        self.yamnet_embed_dir = conf.yamnet_embed_dir

    def get_segment_audio(self, id):
        """Get the audio for a specific segment id."""
        path, start = self.id_to_path_and_start(id)
        audio = self.get_audio_segment(path, start)
        return audio

    def id_to_path_and_start(self, id):
        """Get the audio file path and start for a given segment id."""
        episode_uri, segment_start = id.split("_")
        episode_uri = "spotify:episode:" + episode_uri
        episode = self.metadata[(self.metadata.episode_uri == episode_uri).values]
        audio_file = find_paths(episode, self.audio_path, ".ogg")
        return audio_file[0], int(segment_start)

    def get_paths_and_starts(self, segments):
        """Get the audio file paths for the given segments."""
        paths = []
        starts = []
        for segment in segments:
            path, start = self.id_to_path_and_start(segment["seg_id"])
            paths.append(path)
            starts.append(start)
        return paths, starts
    
    def get_audio_segment(self, path, start, duration=120):
        """Get the required podcast audio segment."""
        waveform, sr = sf.read(
            path,
            start=start * self.sample_rate,
            stop=(start + duration) * self.sample_rate,
            dtype=np.int16,
        )
        if sr != self.sample_rate:
            raise ValueError("Sample rate does not match the required value.")
        waveform = np.mean(waveform, axis=1) / 32768.0
        return waveform
    
    def get_yamnet_embeds(self, segment_id):
        print("TODO: get yamnet embeds")
        ep_prefix, timestamp = segment_id.split("_")
        show_prefix = metadata.loc[metadata['episode_filename_prefix'] == ep_prefix]['show_filename_prefix'].item()
        filepath = relative_file_path(show_prefix, ep_prefix)

        full_path = os.path.join(self.yamnet_embed_dir, filepath + ".h5")
        start_time = float(timestamp)
        end_time = float(timestamp) + 120
        yamnet_embedding = pd.read_hdf(embed_path)
        inbetween_yamnet_embeds = yamnet_embedding[yamnet_embedding.index.to_series().between(start_time, end_time)]

        return inbetween_yamnet_embeds

def clean_text(text):
    """Clean the text to remove non-topical content.

    This includes things like episode numbers, advertisements, and links.
    """
    def isNaN(string):
        return string != string

    # For now just check it is not NaN
    if isNaN(text):
        text = ""

    return text



def topic_task_embeddings(transcripts_path, metadata_subset, save_path='output'):
    seg_length=120
    seg_step=60

    # f = h5py.File(save_path, "w")  

    # See if there are any failed segments
    # failed_uris = None
    # try:
    #     with open("index_failed.txt", "r") as failed_file:
    #         failed_uris = [line.rstrip() for line in failed_file]
    # except Exception:
    #     pass

    # Open file to write failed uri's to
    # failed_file = open("index_failed.txt", "w")

    delcount = 0
    for index, row in tqdm(metadata_subset.iterrows()):
        
        delcount += 1
        print("[del] count: ", delcount)

        if delcount > 1:
            print("gonna break")
            break

        # In try uiteindelijk?
        filename = str(row["episode_filename_prefix"])
        cur_save_file = os.path.join(save_path, filename + '.h5')

        delete_this = True
        if not Path(cur_save_file).exists() or delete_this:
            print("[del] file did not exist, create it! ")
            try:
                print("][del] transpath: ", transcripts_path)
                f = h5py.File(cur_save_file, "w")  
                transcript_path = os.path.join(
                        transcripts_path,
                        src.data.relative_file_path(
                            row["show_filename_prefix"], row["episode_filename_prefix"]
                        )
                        + ".json",
                    )

                seg_base = os.path.splitext(os.path.basename(transcript_path))[0] + "_"
                
                # Get the transcript and find out how long it is
                transcript = src.data.retrieve_timestamped_transcript(transcript_path)       
                last_word_time = math.ceil(transcript["starts"][-1])

                # Generate the segments from the start to the end of the podcasrt
                count = 0
                words_so_far = 0
                for seg_start in range(0, last_word_time, seg_step):
                    count += 1
                    # if count > 3:
                    #     break

                    # Generate the segment name 
                    seg_id = seg_base + str(seg_start)

                    # Find the words in the segment
                    word_indices = np.where(
                        np.logical_and(
                            transcript["starts"] >= seg_start,
                            transcript["starts"] <= seg_start + seg_length,
                        )
                    )[0]

                    word_idx_start_segment = np.where(transcript["starts"] >= seg_start)[0][0]
                    word_idx_stop_segment = np.where(transcript["starts"] <= seg_start + seg_length)[0][-1]

                    seg_words = transcript["words"][word_indices]
                    seg_words = " ".join(seg_words)

                    # Find the number of speakers in the segments
                    seg_speakers = transcript["speaker"][word_indices]
                    num_speakers = len(np.unique(seg_speakers))
                    
                    ## Retreiving text embeddings
                    # Mean of embeds
                    sentences = tokenize.sent_tokenize(seg_words)
                    tokenized_text = indexer.tokenize_text(sentences)
                    embeds = indexer.get_embeds(tokenized_text)
                    text_mean_embed = np.mean(embeds, axis=0)
                    
                    ## Retreiving audio embeddings
                    audio_embed, method = indexer.get_yamnet_embeds(seg_id)
                    
                    grp = f.create_group(seg_id)
                    grp.create_dataset('text_embed', data=text_mean_embed, dtype=np.float32)
                    grp.create_dataset('audio_embed', data=audio_embed, dtype=np.float32)

                    grp.create_dataset("seg_words", data=np.array(seg_words, dtype=h5py.special_dtype(vlen=str)))
                    grp.attrs['num_speakers'] = num_speakers
                    grp.attrs['audio_method'] = method

                    # Also calculate embeddings on sentencelevel
                    grp2 = grp.create_group('sentencelevel')
                    
                    start_index = word_idx_start_segment
                    for idx, embed in enumerate(embeds):
                        # print("sentenclevel ", idx)
                        # if idx > 3:
                        #     break
                        cur_seg_id = seg_id + '_' + str(idx)
                        sent_words = sentences[idx]
                        sent_time_start = transcript['starts'][start_index]
                        last_word_index = start_index +  len(sent_words.split()) - 1

                        # TODO: remove this if statement, it was for DEBUG
                        if last_word_index > len(transcript['starts']) - 1:
                            print("Adjusting extra_words from {} to {}".format(last_word_index, len(transcript['starts'])))
                            last_word_index = len(transcript['starts']) -1
 
                        sent_time_stop = transcript['starts'][last_word_index]

                        # Get yamnet embeds for current timestamps
                        sent_inbetween_yamnet_embeds = indexer.get_yamnets_from_timestamps(sent_time_start, sent_time_stop, seg_id)

                        if len(sent_inbetween_yamnet_embeds) > 0:
                            # Add yamnet embeds and text embeddings for the current sentence to database
                            sent_yamnet_embeds, audio_type = indexer.process_yamnet_embeds(sent_inbetween_yamnet_embeds)
                            grp_slvl = grp2.create_group(cur_seg_id)
                            grp_slvl.create_dataset("sent_words", data=np.array(sent_words, dtype=h5py.special_dtype(vlen=str)))
                            grp_slvl.create_dataset("audio", data=np.array(sent_yamnet_embeds, dtype=np.float32))
                            grp_slvl.create_dataset("text", data=np.array(embed, dtype=np.float32))

                        start_index = last_word_index
                f.close()

            except Exception as e:
                print("Something went wrong! Exception: ", e)
                print(filename)

        else:
            print("Output already exists: ", cur_save_file)

 

def read_metadata_subset(conf, traintest='test'):
    """
    Read the subset of metadata which we will use for the topic task
    NOTE: same func as in index_....py. Move to utils
    """
    metadata = src.data.load_metadata(conf.dataset_path)
    print("[main] metadata loaded ", len(metadata))

    target_dir = os.path.join(conf.dataset_path, 'topic_task')
    colnames = ['num', 'unk', 'episode_uri_time', 'score']

    if traintest == 'test':
        # Read target data
        topics_df_path = os.path.join(target_dir, 'podcasts_2020_topics_test.xml')
        
        ## Read dataframe with 8 query topics
        topics_df = pd.read_xml(topics_df_path)
        print("[main] topics_test loaded ", len(topics_df))

        ## Read annotations
        ## NOTE: line deliminter is different for train and test set
        topics_df_targets_path = os.path.join(target_dir, '2020_test_qrels.list.txt')
        #topics_test_targets = pd.read_csv(topics_test_targets_path, sep='\t', lineterminator='\n', names=colnames)
        topics_df_targets = pd.read_csv(topics_df_targets_path, delimiter=r"\s+", lineterminator='\n', names=colnames)
        print("[main] topics_df_targets loaded for test set", len(topics_df_targets))

    if traintest == 'train':
        # Read target data
        topics_df_path = os.path.join(target_dir, 'podcasts_2020_topics_test.xml')
        
        ## Read dataframe with 8 query topics
        topics_df = pd.read_xml(topics_df_path)

        ## Read annotations
        ## NOTE: line deliminter is different for train and test set
        topics_df_targets_path = os.path.join(target_dir, '2020_train_qrels.list.txt')
        topics_df_targets = pd.read_csv(topics_df_targets_path, sep='\t', lineterminator='\n', names=colnames)
        # topics_test_targets = pd.read_csv(topics_test_targets_path, delimiter=r"\s+", lineterminator='\n', names=colnames)
        print("[main] topics_df_targets loaded for train set", len(topics_df_targets))

    # remove float '.0' from column string
    topics_df_targets['episode_uri_time'] = topics_df_targets['episode_uri_time'].str.replace(r'.0$', '', regex=True)

    # Add a binary score (not ranked)
    ## scores [1,2,3,4] will be considered relevant
    ## scores [0 will be consisdered as irrelevant]
    topics_df_targets['bin_score'] = topics_df_targets['score'] > 0
    topics_df_targets.bin_score.replace((True, False), (1, 0), inplace=True)

    # Add column with only episode_uri and only_time
    topics_df_targets[['episode_uri','time']] = topics_df_targets.episode_uri_time.str.split(pat='_',expand=True)

    # Remove all podcasts that do not have an annotation
    metadata_subset = pd.merge(metadata, topics_df_targets, how='left', indicator='Exist')
    metadata_subset['Exist'] = np.where(metadata_subset.Exist == 'both', True, False)
    metadata_subset = metadata_subset[metadata_subset['Exist']==True].drop(['Exist','episode_uri'], axis=1)
    print("[main] metadata_subset loaded ", len(metadata_subset))

    # remove 'spotify:episode:' from column so we can match items
    topics_df_targets['episode_uri_time'] = topics_df_targets.episode_uri_time.str.replace(r'spotify:episode:', '')
    topics_df_targets['episode_uri'] = topics_df_targets.episode_uri_time.str.replace(r'spotify:episode:', '')

    # Remove timestamps from episode uri
    topics_df_targets['episode_uri'] = topics_df_targets['episode_uri'].str.split('_').str[0]

    # TODO: DELETE THESE TWO LINES
    metadata_subset = metadata_subset.sort_values('score')

    return metadata_subset, topics_df, topics_df_targets


if __name__ == "__main__":
    print("[main]")
    """TO ARGS"""
    model_path = '/Users/casper/Documents/UvAmaster/b23456_thesis/msc_thesis/code/contrastive_mm2/logs/load_test'
    # RNN
    model_path = '/Users/casper/Documents/UvAmaster/b23456_thesis/msc_thesis/code/contrastive_mm2/logs/lisa_v2-simcse_loss_rnn_relu_768_0.001_2022-05-12_15-58-03'
    # model_path = "E:\msc_thesis\code\contrastive_mm2\logs\lisa_v2-simcse_loss_rnn_relu_768_0.001_2022-05-12_15-58-03"
    model_path = "E:\msc_thesis\code\contrastive_mm2\logs\lisa_v2-simcse_loss_rnn_relu_768_2e-05_2022-05-17_06-58-44"
    model_path = '/Users/casper/Documents/UvAmaster/b23456_thesis/msc_thesis/code/contrastive_mm2/logs/windows_gru2-clip_loss_gru_gelu_768_5e-05_2022-05-26_21-51-25'

    # SIMPLE PROJ head
    # model_path = '/Users/casper/Documents/UvAmaster/b23456_thesis/msc_thesis/code/contrastive_mm2/logs/lisa_v2-simcse_loss_simple_projection_head_relu_768_2e-05_2022-05-12_15-56-09'

    transcripts_path = '/Users/casper/Documents/UvAmaster/b23456_thesis/msc_thesis/code/data/sp/podcasts-no-audio-13GB/podcasts-transcripts'
    # transcripts_path = 'E:/msc_thesis/code/data/sp/podcasts-no-audio-13GB/podcasts-transcripts'

    # Whether we create for the train or the tes split. 
    traintest = 'test'

    # Creating output directory.
    save_name = os.path.split(model_path)[-1]
    save_path = os.path.join(conf.yamnet_topic_embed_path, save_name + "_" + traintest)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Reading the metadata.
    metadata_subset, topics_df, topics_df_targets = read_metadata_subset(conf, traintest='test')

    # Remove duplicate rows
    print("[del]: worden heir wel alle dulicates echt verwijderd? zitten er timestamps bij episode_filename_prefix?")
    metadata_subset = metadata_subset.drop_duplicates(subset=['episode_filename_prefix']).sort_values('episode_filename_prefix')
    print("[main] Metadata length: ", len(metadata_subset))

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
    # tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    print("[Load model] model loaded: ", model_weights)
    print(full_model.state_dict()['audio_model.0.seq_model.weight_ih_l1'][0][0])
    
    # Create h5py files with embeddings for all episodes in the test-set
    indexer = Indexer(full_model, tokenizer, CFG, config_path=config_path)
    topic_task_embeddings(transcripts_path, metadata_subset, save_path=save_path)
    print("[main] done")



            



