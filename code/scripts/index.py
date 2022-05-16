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

import src.data

from src.data import load_metadata, find_paths, relative_file_path
from clipmodel.gru_test import CLIPModel

# Load static configuration variables. 
config_path = "./config.yaml"
conf = OmegaConf.load(config_path)
print("[cudacheck] Is cuda available? ", torch.cuda.is_available())


@dataclass_json
@dataclass
class FullCfg:
    batch_size: int
    loss_type: str
    audio_proj_head: str
    audio_activation: str
        
    text_proj_head: str
        
    final_projection_dim: int
    debug = False
    head_lr = 1e-3
    audio_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_encoder_input = 1024
    audio_hidden_dim = 768
    layer_dim = 3

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
        
    def tokenize_text(self, sentences):
        # Tokenize text
        tokenized_text = self.tokenizer(
            sentences, padding=True, truncation=True, max_length=200, return_tensors='pt', return_token_type_ids=True,
        )
        return tokenized_text

    def get_embeds(self, tokenized_text):
        input_ids = tokenized_text['input_ids'].to(self.device)
        attention_mask = tokenized_text['attention_mask'].to(self.device)

        with torch.no_grad():
            embeds = self.model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeds = embeds.detach().cpu().numpy()

        return embeds

    def get_yamnets_from_timestamp(self, segment_id):
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
        if type(self.model.audio_encoder).__name__ == 'simple_ProjectionHead':
            audio_type = 'mean'
            lengths = None
            mean_embeds = torch.tensor(inbetween_yamnet_embeds.mean()).to(torch.float32)

            mean_embeds = mean_embeds.reshape(1,-1)
            
            self.model.eval()
            with torch.no_grad():
                embeds = self.model.audio_encoder(mean_embeds, lengths)
            return embeds[0], audio_type
        elif type(self.model.audio_encoder).__name__ == 'SequentialAudioModel':
            #print("TODO: GRU GET_YAMNET_EMBEDS")
            audio_type = 'gru'
            x = torch.tensor(inbetween_yamnet_embeds.values).to(torch.float32)
            x = x.to(self.device)
            h0 = torch.zeros(self.layer_dim, 1, self.audio_hidden_dim).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                x = x[None,:,:]
                # embedded = torch.nn.utils.rnn.pack_padded_sequence(audio_embeds, length, batch_first=True, enforce_sorted=False)
                out, h0 = self.model.audio_encoder.seq_model(x, h0.detach())
                
                test = h0[-1, :, :]
                out = self.model.audio_encoder.fc(test)
                
            #embeds = self.model.audio_encoder(inbetween_yamnet_embeds)
            return out[0], audio_type

    
    def get_yamnet_embeds(self, segment_id):
        inbetween_yamnet_embeds = self.get_yamnets_from_timestamp(segment_id)
        
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


class PodcastSegment(Document):

    """Implementation of a podcast segment document in elasticsearch."""
    show_name = Text(analyzer="snowball")
    show_desc = Text(analyzer="snowball")
    epis_name = Text(analyzer="snowball")
    epis_desc = Text(analyzer="snowball")
    seg_words = Text(analyzer="snowball")
    seg_length = Integer()
    seg_speakers = Integer()

    #fake_embed = DenseVector(dims=mutual_embed_dim, index=True, similarity="l2_norm")
    mutual_embed_dim = 768 ## TODO: FIX THIS
    #text_embed = DenseVector(dims=mutual_embed_dim, index=True, similarity="l2_norm")
    text_mean_embed = DenseVector(dims=mutual_embed_dim, index=True, similarity="l2_norm")
    text_begin_embed = DenseVector(dims=mutual_embed_dim, index=True, similarity="l2_norm")
    
    audio_proj_head_embed = DenseVector(dims=mutual_embed_dim, index=True, similarity="l2_norm")
    audio_gru_embed = DenseVector(dims=mutual_embed_dim, index=True, similarity="l2_norm")
        

    class Index:
        """Elasticsearch index definition."""
        name = "segments"

    def save(self, **kwargs):
        """Save the document to Elasticsearch."""
        self.seg_length = len(self.seg_words.split())
        return super(PodcastSegment, self).save(**kwargs)
def init_index():
    """Set up the Elasticsearch index by creating the mappings."""
    PodcastSegment.init()



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


def add_podcast(
    searcher, 
    indexer,
    transcript_path,
    show_name,
    show_desc,
    epis_name,
    epis_desc,
    seg_length=120,
    seg_step=60,
    new_embed = None
):
    """Get podcast transcript data to be indexed."""
    #print("[add_pocast] ", fake_embed)

    # Generate the segment basename
    seg_base = os.path.splitext(os.path.basename(transcript_path))[0] + "_"
    #print("seg_base: ", seg_base)
    
    # Clean the show and episode names and descriptions
    show_name = clean_text(show_name)
    show_desc = clean_text(show_desc)
    epis_name = clean_text(epis_name)
    epis_desc = clean_text(epis_desc)

    # Get the transcript and find out how long it is
    transcript = src.data.retrieve_timestamped_transcript(transcript_path)
    
    last_word_time = math.ceil(transcript["starts"][-1])
    #print("lastword: ", last_word_time)

    # Generate the segments from the start to the end of the podcasrt
    count = 0
    for seg_start in range(0, last_word_time, seg_step):
        #print("will add ", fake_embeds[count][0])
        #print("segstart: ", seg_start)
        # Generate the segment name
        seg_id = seg_base + str(seg_start)

        # Find the words in the segment
        word_indices = np.where(
            np.logical_and(
                transcript["starts"] >= seg_start,
                transcript["starts"] <= seg_start + seg_length,
            )
        )[0]
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
        text_mean_embeds = np.mean(embeds, axis=0)

        # Embeds based on first 512 chars
        tokenized_text_full = indexer.tokenize_text(seg_words)
        text_begin_embed = indexer.get_embeds(tokenized_text_full)[0]
        
        ## Retreiving audio embeddings
        # TODO!!
        audio_embeds, method = indexer.get_yamnet_embeds(seg_id)
        if method == 'mean':
            audio_proj_head_embed = audio_embeds.tolist()
            #audio_gru_embed = torch.zeros_like(audio_embeds).tolist()
            segment = PodcastSegment(
                meta={"id": seg_id},
                show_name=show_name,
                show_desc=show_desc,
                epis_name=epis_name,
                epis_desc=epis_desc,
                seg_words=seg_words,
                seg_speakers=num_speakers,
                mean_embed=text_mean_embeds.tolist(),
                begin_embed = text_begin_embed.tolist(),
                audio_proj_head_embed = audio_proj_head_embed
            )
        else:
            audio_gru_embed = audio_embeds.tolist()
            print("[del] gru for : ", seg_words)
            #audio_proj_head_embed = torch.zeros_like(audio_embeds)
            segment = PodcastSegment(
                meta={"id": seg_id},
                show_name=show_name,
                show_desc=show_desc,
                epis_name=epis_name,
                epis_desc=epis_desc,
                seg_words=seg_words,
                seg_speakers=num_speakers,
                mean_embed=text_mean_embeds.tolist(),
                begin_embed = text_begin_embed.tolist(),
                audio_gru_embed = audio_gru_embed
            )
            
        try:
            segment.save()
            #print("saved")
        except Exception as e:
            raise ConnectionError("Indexing error: {}".format(e))
            
        count += 1
        # for now ignore per_sentence
        continue
        for idx, embed in enumerate(embeds):
            cur_seg_id = seg_id + '_' + str(idx)
            print("Also add id: ", cur_seg_id)

            seg_words = sentences[idx]
            # Create and save the segment per sentence
            segment = PodcastSegment(
                meta={"id": cur_seg_id},
                show_name=show_name,
                show_desc=show_desc,
                epis_name=epis_name,
                epis_desc=epis_desc,
                seg_words=seg_words,
                seg_speakers=num_speakers,
                new_embed=embed.tolist()
            )
            try:
                segment.save()
                #print("saved")
            except Exception as e:
                raise ConnectionError("Indexing error: {}".format(e))
            

def add_df_to_elasticsearch(transcripts_path, metadata_subset):

    # See if there are any failed segments
    failed_uris = None
    try:
        with open("index_failed.txt", "r") as failed_file:
            failed_uris = [line.rstrip() for line in failed_file]
    except Exception:
        pass

    # Open file to write failed uri's to
    failed_file = open("index_failed.txt", "w")

    delcount = 0
    for index, row in tqdm(metadata_subset.iterrows()):
        delcount += 1
        # print("delcount: ", delcount)
        if delcount > 560:
            break
        if delcount < 300:
            print("continue")
            continue

        if (failed_uris and str(row["episode_filename_prefix"]) in failed_uris) or not failed_uris:
            transcript_path = os.path.join(
                transcripts_path,
                src.data.relative_file_path(
                    row["show_filename_prefix"], row["episode_filename_prefix"]
                )
                + ".json",
            )

            # add_podcast(
            #     searcher,
            #     indexer,
            #     transcript_path,
            #     row["show_name"],
            #     row["show_description"],
            #     row["episode_name"],
            #     row["episode_description"],
            # )

            try:
                add_podcast(
                    searcher,
                    indexer,
                    transcript_path,
                    row["show_name"],
                    row["show_description"],
                    row["episode_name"],
                    row["episode_description"],
                )
                
            except Exception as e:
                print("EXCEPTION: ", e)
                pass
                failed_file.write(str(row["episode_filename_prefix"]) + "\n")
    # Close failed file
    failed_file.close()
 
if __name__ == "__main__":
    print("[main]")
    """TO ARGS"""
    model_path = '/Users/casper/Documents/UvAmaster/b23456_thesis/msc_thesis/code/contrastive_mm2/logs/load_test'
    # RNN
    model_path = '/Users/casper/Documents/UvAmaster/b23456_thesis/msc_thesis/code/contrastive_mm2/logs/v2-simcse_loss_rnn_relu_768_0.001_2022-05-12_15-58-03'
    # SIMPLE PROJ head
    # model_path = '/Users/casper/Documents/UvAmaster/b23456_thesis/msc_thesis/code/contrastive_mm2/logs/v2-simcse_loss_simple_projection_head_relu_768_2e-05_2022-05-12_15-56-09'

    transcripts_path = '/Users/casper/Documents/UvAmaster/b23456_thesis/msc_thesis/code/data/sp/podcasts-no-audio-13GB/podcasts-transcripts'

    """READ METADATA (new func)"""
    # Read metadata
    metadata = src.data.load_metadata(conf.dataset_path)
    print("[main] metadata loaded ", len(metadata))

    # Read target data
    target_dir = os.path.join(conf.dataset_path, 'topic_task')
    topics_train_path = os.path.join(target_dir, 'podcasts_2020_topics_train.xml')
    
    ## Read dataframe with 8 query topics
    topics_train = pd.read_xml(topics_train_path)
    print("[main] topics_train loaded ", len(topics_train))

    ## Read annotations
    colnames = ['num', 'unk', 'episode_uri_time', 'score']
    topics_train_targets_path = os.path.join(target_dir, '2020_train_qrels.list.txt')
    topics_train_targets = pd.read_csv(topics_train_targets_path, sep='\t', lineterminator='\n', names=colnames)
    print("[main] topics_train_targets loaded ", len(topics_train_targets))

    # remove float '.0' from column string
    topics_train_targets['episode_uri_time'] = topics_train_targets['episode_uri_time'].str.replace(r'.0$', '', regex=True)

    # Add a binary score (not ranked)
    ## scores [1,2,3,4] will be considered relevant
    ## scores [0 will be consisdered as irrelevant]
    topics_train_targets['bin_score'] = topics_train_targets['score'] > 0
    topics_train_targets.bin_score.replace((True, False), (1, 0), inplace=True)

    # Add column with only episode_uri and only_time
    topics_train_targets[['episode_uri','time']] = topics_train_targets.episode_uri_time.str.split(pat='_',expand=True)

    # Remove all podcasts that do not have an annotation
    metadata_subset = pd.merge(metadata, topics_train_targets, how='left', indicator='Exist')
    metadata_subset['Exist'] = np.where(metadata_subset.Exist == 'both', True, False)
    metadata_subset = metadata_subset[metadata_subset['Exist']==True].drop(['Exist','episode_uri'], axis=1)
    print("[main] metadata_subset loaded ", len(metadata_subset))

    # remove 'spotify:episode:' from column so we can match items
    topics_train_targets['episode_uri_time'] = topics_train_targets.episode_uri_time.str.replace(r'spotify:episode:', '')
    topics_train_targets['episode_uri'] = topics_train_targets.episode_uri_time.str.replace(r'spotify:episode:', '')

    # TODO: DELETE THESE TWO LINES
    metadata_subset = metadata_subset.sort_values('score')
    # metadata_subset = metadata_subset.iloc[[550,595,605,608]]
    # metadata_subset = metadata_subset
    # print(len(metadata_subset))
    # exit(1)


    """LOAD MODEL (new func)"""
    print("[Load model] from ", model_path)
    model_weights = os.path.join(model_path, "/output/full_model_weights.pth")
    print("[Load model] weights loaded")
    model_config_path = os.path.join(model_path, 'config.json')
    
    # Opening JSON file
    f = open(model_config_path)
    model_config = json.load(f)
    CFG = from_dict(data_class=FullCfg, data=model_config)
    mutual_embed_dim = CFG.final_projection_dim
    print("[Load model] config loaded: ", CFG)
    

    # Load model and tokenizer
    full_model = CLIPModel(CFG)
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    print("[Load model] model loaded: ")
    
    indexer = Indexer(full_model, tokenizer, CFG, config_path=config_path)

    # Define client connection and setup index
    connections.create_connection(hosts=["localhost"])
    init_index()

    searcher = Searcher()
    print("[main] Add searcher loaded")

    print("[main] Add df to elasticsearch")
    add_df_to_elasticsearch(transcripts_path, metadata_subset)
    print("[main] done")
    




            



