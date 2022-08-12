"""
Contrastive multimodal learning
Author: Casper Wortmann
Usage: python main.py
"""
import os
from re import sub 

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import json 
import torch
import h5py
import math

import sys
sys.path.append('../') 
from utils import get_embed_transcript_paths, read_metadata_subset, find_paths
from omegaconf import OmegaConf

def save_all(save_path, all_mean_embeddings, filenames, in_train_set, all_sentences, all_full_embeddings, segment_tsps):
    # Sort everything according to sentence level
    lengths = np.array([len(s) for s in all_sentences])
    inds = lengths.argsort()

    all_sentences = [all_sentences[i] for i in inds]
    all_mean_embeddings = [all_mean_embeddings[i] for i in inds]
    all_full_embeddings = [all_full_embeddings[i] for i in inds]
    filenames = [filenames[i] for i in inds]
    segment_tsps = [segment_tsps[i] for i in inds]

    comp_lvl = 9
    f = h5py.File(save_path, "w")  
    all_mean_embeddings = torch.stack(all_mean_embeddings, dim=0)

    # Create a new group to the dataset and add embeddings and sentences.
    f.create_dataset("mean_embeddings", data=all_mean_embeddings, compression="gzip", compression_opts=comp_lvl)
    f.create_dataset("sentences", data=np.array(all_sentences, dtype=h5py.special_dtype(vlen=str)))

    seg_ts = [filenames[i] + '_' + str(segment_tsps[i]) for i in range(len(filenames))]
    f.create_dataset("seg_ts", data=np.array(seg_ts, dtype=h5py.special_dtype(vlen=str)))

    max_embed_dim = 0
    for embed_idx, full_embeds in enumerate(all_full_embeddings):
        f.create_dataset(str(embed_idx), data=full_embeds, compression="gzip", compression_opts=comp_lvl)
        max_embed_dim = max(max_embed_dim, len(full_embeds))
    f.attrs['max_embed_dim'] = max_embed_dim
    f.close()

if __name__ == "__main__":
    # Setup configuration file. 
    conf = OmegaConf.load("../config.yaml")
    conf.dataset_path = os.path.join("../", conf.dataset_path)
    conf.sp_train_test_split = os.path.join("../", conf.sp_train_test_split)

    # Load all the metadata.
    metadata = pd.read_csv(conf.dataset_path + "metadata.tsv", delimiter="\t")
    print("[main] total length of metadata: ", len(metadata))

    # Reading the metadata.
    metadata_testset, topics_df, topics_df_targets = read_metadata_subset(conf, traintest='test')

    # Remove duplicate rows.
    metadata_testset = metadata_testset.drop_duplicates(subset=['episode_filename_prefix']).sort_values('episode_filename_prefix')

    # Exclude the testset from the metadata.
    metadata = pd.merge(metadata, metadata_testset, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

    # Check how many yamnet embeddings are stored on local disk.
    subset_shows = [h5file.parts[-2] for h5file in Path(conf.yamnet_embed_path).glob('**/*.h5')]
    subset_shows = set(subset_shows)
    subset = metadata[(metadata['show_filename_prefix'].isin(subset_shows))]
    print("[main] number of data found on local disk: ", len(subset)) 
    
    # Determine train/test set
    tmp = subset.drop_duplicates(subset='show_filename_prefix', keep="last")
    train = tmp.sample(frac=0.95, random_state=200) #random state is a seed value
    train_shownames = train.show_filename_prefix.tolist()
    test = tmp.drop(train.index)
    subset["set"] = np.where(subset["show_filename_prefix"].isin(train_shownames), "train", "val")
    dataframe =  metadata_testset.append(subset)

    #  Determine the test set
    dataframe.set.isna().sum() 
    dataframe.set = dataframe.set.fillna('test')

    # Store train/val/test split on local disk
    train_split = dict(zip(dataframe.episode_filename_prefix, dataframe.set))
    train_split_filename = conf.sp_train_test_split
    with open(train_split_filename, 'w') as fp:
        json.dump(train_split, fp, sort_keys=True, indent=4)

    # Read transcripts
    transcripts_dir = os.path.join(conf.dataset_path, 'podcasts-transcripts')
    transcripts_paths = find_paths(dataframe, transcripts_dir, ".json")
    transcripts_paths = sorted(transcripts_paths)
    outer_folders, e_filenames, t_filenames = get_embed_transcript_paths(transcripts_paths)

    # Create output folders
    Path(os.path.join(conf.yamnet_processed_path, "train")).mkdir(parents=True, exist_ok=True) 
    Path(os.path.join(conf.yamnet_processed_path, "val")).mkdir(parents=True, exist_ok=True) 
    Path(os.path.join(conf.yamnet_processed_path, "test")).mkdir(parents=True, exist_ok=True) 

    last_output_folder = outer_folders[0]
    train_all_sentences = []
    train_all_full_embeddings = []
    train_all_mean_embeddings = []
    train_filenames = []
    train_in_train_set = []
    train_segment_tsps = []

    val_all_sentences = []
    val_all_full_embeddings = []
    val_all_mean_embeddings = []
    val_filenames = []
    val_in_train_set = []
    val_segment_tsps = []

    test_all_sentences = []
    test_all_full_embeddings = []
    test_all_mean_embeddings = []
    test_filenames = []
    test_in_train_set = []
    test_segment_tsps = []

    save_every = 9  # We save every 9 outer folders. That way the size remains manageable. 
    save_count = 0
    for idx in range(len(outer_folders)):
        outer_folder = outer_folders[idx]
        filename = os.path.split(e_filenames[idx])[-1].split('.')[0]

        if outer_folder != last_output_folder:
            save_count += 1

            if outer_folder[0] != last_output_folder[0]:
                
                save_count = 0
                save_path_train = os.path.join(conf.yamnet_processed_path, "train", last_output_folder.replace(os.sep, '_') + '_embeds.h5')
                save_path_val = os.path.join(conf.yamnet_processed_path, "val", last_output_folder.replace(os.sep, '_') + '_embeds.h5')
                save_path_test = os.path.join(conf.yamnet_processed_path, "test", last_output_folder.replace(os.sep, '_') + '_embeds.h5')

                if len(train_filenames) > 0:
                    save_all(save_path_train, 
                            train_all_mean_embeddings, 
                            train_filenames, 
                            train_in_train_set, 
                            train_all_sentences, 
                            train_all_full_embeddings,
                            train_segment_tsps)
                if len(val_filenames) > 0:
                    save_all(save_path_val, 
                            val_all_mean_embeddings, 
                            val_filenames, 
                            val_in_train_set, 
                            val_all_sentences, 
                            val_all_full_embeddings,
                            val_segment_tsps)

                if len(test_filenames) > 0:
                    save_all(save_path_test, 
                            test_all_mean_embeddings, 
                            test_filenames, 
                            test_in_train_set, 
                            test_all_sentences, 
                            test_all_full_embeddings,
                            test_segment_tsps)

                train_all_sentences = []
                train_all_full_embeddings = []
                train_all_mean_embeddings = []
                train_filenames = []
                train_in_train_set = []
                train_segment_tsps = []
                val_all_sentences = []
                val_all_full_embeddings = []
                val_all_mean_embeddings = []
                val_filenames = []
                val_in_train_set = []
                val_segment_tsps = []
                test_all_sentences = []
                test_all_full_embeddings = []
                test_all_mean_embeddings = []
                test_filenames = []
                test_in_train_set = []
                test_segment_tsps = []

            last_output_folder = outer_folder

        if idx % 10 == 0:
            print("\t processing {}/{}".format(idx, len(outer_folders)))

        # Load yamnet embeddings and scores
        embed_path = os.path.join(conf.yamnet_embed_path, outer_folder, e_filenames[idx])
        yamnet_embedding = pd.read_hdf(embed_path)

        # Load the transcript
        transcript_path = os.path.join(transcripts_dir, outer_folder, t_filenames[idx])
        transcript_json = load_transcript(transcript_path)

        # Extract all sentences and corresponding words
        sentences, timestamps, segment_ts, mean_embeddings, full_embeddings = extract_transcript(transcript_json, yamnet_embedding)

        if train_split[filename] == 'train':
            train_all_sentences.extend(sentences)
            train_all_mean_embeddings.extend(mean_embeddings)
            train_all_full_embeddings.extend(full_embeddings)
            
            # Check if it belongs to train or test split.
            train_filenames.extend([filename] * len(sentences))
            train_in_train_set.extend([train_split[filename]] * len(sentences))
            
            # Add timestamps.
            train_segment_tsps.extend(segment_ts)
            
        elif train_split[filename] == 'val':
            val_all_sentences.extend(sentences)
            val_all_mean_embeddings.extend(mean_embeddings)
            val_all_full_embeddings.extend(full_embeddings)

            # Check if it belongs to train or test split.
            val_filenames.extend([filename] * len(sentences))
            val_in_train_set.extend([train_split[filename]] * len(sentences))
            
            # Add timestamps.
            val_segment_tsps.extend(segment_ts)

        elif train_split[filename] == 'test':
            test_all_sentences.extend(sentences)
            test_all_mean_embeddings.extend(mean_embeddings)
            test_all_full_embeddings.extend(full_embeddings)

            # Check if it belongs to train or test split.
            test_filenames.extend([filename] * len(sentences))
            test_in_train_set.extend([train_split[filename]] * len(sentences))
            
            test_segment_tsps.extend(segment_ts)

    save_path_train = os.path.join(conf.yamnet_processed_path, "train", last_output_folder.replace(os.sep, '_') + '_embeds.h5')
    save_path_val = os.path.join(conf.yamnet_processed_path, "val", last_output_folder.replace(os.sep, '_') + '_embeds.h5')
    save_path_test = os.path.join(conf.yamnet_processed_path, "test", last_output_folder.replace(os.sep, '_') + '_embeds.h5')
    print("saving {}/{}".format(idx, len(outer_folders)))

    if len(train_filenames) > 0:
        save_all(save_path_train, 
                train_all_mean_embeddings, 
                train_filenames, 
                train_in_train_set, 
                train_all_sentences, 
                train_all_full_embeddings,
                train_segment_tsps)
    if len(val_filenames) > 0:
        save_all(save_path_val, 
                val_all_mean_embeddings, 
                val_filenames, 
                val_in_train_set, 
                val_all_sentences, 
                val_all_full_embeddings,
                val_segment_tsps)
        
    if len(test_filenames) > 0:
        save_all(save_path_test, 
                test_all_mean_embeddings, 
                test_filenames, 
                test_in_train_set, 
                test_all_sentences, 
                test_all_full_embeddings,
                test_segment_tsps)

