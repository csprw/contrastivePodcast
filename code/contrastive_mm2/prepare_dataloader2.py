"""
Contrastive multimodal learning
Author: Casper Wortmann
Usage: python main.py
"""
import os
import pickle
from re import sub 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import json 
from nltk import tokenize
import torch
import h5py

import sys
sys.path.append('../scripts/') 
from prepare_index_sentencelevel2 import read_metadata_subset

from omegaconf import OmegaConf
conf = OmegaConf.load("./config.yaml")


def load_metadata(dataset_path):
    """
    Note: function from []
    Load the Spotify podcast dataset metadata.
    """
    return pd.read_csv(dataset_path + "metadata.tsv", delimiter="\t")


def find_paths(metadata, base_folder, file_extension):
    """
    Note: function from []
    Find the filepath based on the dataset structure.

    Uses the metadata, the filepath folder and the file extension you want.

    Args:
        metadata (df): The metadata of the files you want to create a path for
        base_folder (str): base directory for where data is (to be) stored
        file_extension (str): extension of the file in the path

    Returns:
        paths (list): list of paths (str) for all files in the given metadata
    """
    paths = []
    for i in range(len(metadata)):
        relative_path = relative_file_path(
            metadata.show_filename_prefix.iloc[i],
            metadata.episode_filename_prefix.iloc[i],
        )
        path = os.path.join(base_folder, relative_path + file_extension)
        paths.append(path)
    return paths

def relative_file_path(show_filename_prefix, episode_filename_prefix):
    """
    Note: function from []
    Return the relative filepath based on the episode metadata.
    """
    return os.path.join(
        show_filename_prefix[5].upper(),
        show_filename_prefix[6].upper(),
        show_filename_prefix,
        episode_filename_prefix,
    )

def load_transcript(path):
    """
    Note: function from []
    Load a python dictionary with the .json transcript.
    """
    with open(path, "r") as file:
        transcript = json.load(file)
    return transcript

def get_sent_indexes(sentences):
    indexes = []
    lb = 0
    ub = 0

    for s in sentences:
        extra_indexes = len(s.split())
        ub += extra_indexes 
        indexes.append((lb, ub-1))
        lb += extra_indexes
    return indexes
#Note: speaker tags are in the final "results"
def extract_transcript(transcript_json, yamnet_embedding):
    '''
    Extracts sentences and corresponding timestamps from a transcript.
    Input: 
        Json file containing a podcast transcript
    Output:
        sentences: a list of sentences
        timestamps: a list of tuples [(starttime, endtime)]
        mean_embeddings: the mean of the yamnet embeddings inbetween (starttime, endtime)
    '''

    sentences = []
    timestamps = []
    segment_timestamps = []
    mean_embeddings = []
    full_embeddings = []
    
    seg_duration = 60.0
    
    for result in transcript_json['results'][:-1]:
        if len(result['alternatives'][0]) == 0:
            #print("Will continue")
            continue
        best_alternative = result['alternatives'][0]

        cur_sentences = tokenize.sent_tokenize(best_alternative['transcript'])

        # Returns a list of tuples, with (starttime, endtime) for each sentence
        indexes = get_sent_indexes(cur_sentences)

        for idx_sentence,  (idx_start, idx_end) in enumerate(indexes):
            start_time = float(best_alternative['words'][idx_start]['startTime'][:-1])
            end_time = float(best_alternative['words'][idx_end]['endTime'][:-1])
            
            # Segment time is the nearest floored multiple of 60
            # This is in line with the topic task at hand.
            segment_time = seg_duration * math.floor(start_time/seg_duration)
            
            full_embedding = yamnet_embedding[yamnet_embedding.index.to_series().between(start_time, end_time)]
            mean_embedding = yamnet_embedding[yamnet_embedding.index.to_series().between(start_time, end_time)].mean()
            
            # If a sentence is too short to contain embeddings, skip it
            if not mean_embedding.isnull().values.any():
                mean_embeddings.append(torch.tensor(mean_embedding))
                sentences.append(cur_sentences[idx_sentence])
                timestamps.append((start_time, end_time)) # To h5py
                segment_timestamps.append(segment_time)
                full_embeddings.append(torch.tensor(full_embedding.values))
    assert len(sentences) == len(timestamps) == len(mean_embeddings)
    return sentences, timestamps, segment_timestamps, mean_embeddings, full_embeddings

def find_paths(metadata, base_folder, file_extension):
    """
    Note: function from []
    Find the filepath based on the dataset structure.

    Uses the metadata, the filepath folder and the file extension you want.

    Args:
        metadata (df): The metadata of the files you want to create a path for
        base_folder (str): base directory for where data is (to be) stored
        file_extension (str): extension of the file in the path

    Returns:
        paths (list): list of paths (str) for all files in the given metadata
    """
    paths = []
    for i in range(len(metadata)):
        relative_path = relative_file_path(
            metadata.show_filename_prefix.iloc[i],
            metadata.episode_filename_prefix.iloc[i],
        )
        path = os.path.join(base_folder, relative_path + file_extension)
        paths.append(path)
    return paths


def get_dirnames(transcripts_dir):
    outer_dirnames = []
    inner_dirnames = []
    for outer_dirname in glob.iglob(transcripts_dir + '/**', recursive=False):
        if os.path.isdir(outer_dirname): # filter dirs
            outer_dirnames.append(os.path.split(outer_dirname)[-1])
            for inner_dirname in glob.iglob(outer_dirname + '/**', recursive=False):
                inner_dirnames.append(os.path.split(inner_dirname)[-1])
    return sorted(outer_dirnames), sorted(inner_dirnames)

def get_embed_transcript_paths(transcripts_paths):
    outer_folders = []
    e_filenames = []
    t_filenames = []
    for p in transcripts_paths:
        splitted_p = os.path.normpath(p).split(os.path.sep)

        outer_folders.append(os.path.join(splitted_p[-4], splitted_p[-3]))

        t_filenames.append(os.path.join(splitted_p[-2], splitted_p[-1]))
        e_filenames.append(os.path.join( splitted_p[-2], splitted_p[-1].split('.')[-2]+".h5"))

    return outer_folders, e_filenames, t_filenames

def save_all(save_path, all_mean_embeddings, filenames, in_train_set, all_sentences, all_full_embeddings, segment_tsps):
    comp_lvl = 9
    f = h5py.File(save_path, "w")  
    all_mean_embeddings = torch.stack(all_mean_embeddings, dim=0)

    # Create a new group to the dataset and add embeddings and sentences.
    #dset0 = f.create_dataset("filenames", data=np.array(filenames, dtype='S'))
    dset1 = f.create_dataset("mean_embeddings", data=all_mean_embeddings, compression="gzip", compression_opts=comp_lvl)
    dset2 = f.create_dataset("sentences", data=np.array(all_sentences, dtype=h5py.special_dtype(vlen=str)))
    #dset3 = f.create_dataset("train_split", data=np.array(in_train_set, dtype='S'))
    #dset3 = f.create_dataset("segment_ts", data=np.array(segment_tsps), compression="gzip", compression_opts=comp_lvl)
    
    seg_ts = filenames + '_' + str(segment_tsps)
    print("del: is this correct? ", seg_ts)
    raise
    dset0 = f.create_dataset("seg_ts", data=np.array(filenames, dtype='S'))

    max_embed_dim = 0
    for embed_idx, full_embeds in enumerate(all_full_embeddings):
        dset = f.create_dataset(str(embed_idx), data=full_embeds, compression="gzip", compression_opts=comp_lvl)
        max_embed_dim = max(max_embed_dim, len(full_embeds))
    f.attrs['max_embed_dim'] = max_embed_dim
    f.close()




if __name__ == "__main__":
    print("[main] reading data from: ")
    print(conf.dataset_input_path)

    # Load all the metadata
    metadata = load_metadata(conf.dataset_input_path)
    # metadata = metadata[:100] # [del] delete!
    print("[main] total length of metadata: ", len(metadata))


    # Load the topic task test set:
    # Reading the metadata.
    metadata_testset, topics_df, topics_df_targets = read_metadata_subset(conf, traintest='test')
    # Remove duplicate rows
    print("[del]: worden heir wel alle dulicates echt verwijderd? zitten er timestamps bij episode_filename_prefix?")
    metadata_testset = metadata_testset.drop_duplicates(subset=['episode_filename_prefix']).sort_values('episode_filename_prefix')
    print("[main] metadata_testset length: ", len(metadata_testset))


    # # Exclude the testset from the metadata
    metadata = pd.merge(metadata, metadata_testset, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    print("Metadata samples for train and validation: ", len(metadata))


    # Check how many yamnet embeddings are stored on local disk
    subset_shows = [h5file.parts[-2] for h5file in Path(conf.yamnet_embed_dir).glob('**/*.h5')]
    subset_shows = set(subset_shows)
    subset = metadata[(metadata['show_filename_prefix'].isin(subset_shows))]
    print("[main] number of data found on local disk: ", len(subset)) 

    
    # Determine train/test set
    tmp = subset.drop_duplicates(subset='show_filename_prefix', keep="last")
    train = tmp.sample(frac=0.55, random_state=200) #random state is a seed value
    train_shownames = train.show_filename_prefix.tolist()
    test = tmp.drop(train.index)
    subset["set"] = np.where(subset["show_filename_prefix"].isin(train_shownames), "train", "val")

    print(subset.head(5))

    dataframe =  metadata_testset.append(subset)
    print(dataframe.head(5))

    #  Determine the test set
    dataframe.set.isna().sum() 
    dataframe.set = dataframe.set.fillna('test')

    # Store train/test split on local disk
    train_split = dict(zip(dataframe.episode_filename_prefix, dataframe.set))
    # print(train_split)
    train_split_filename = conf.sp_train_test_split
    with open(train_split_filename, 'w') as fp:
        json.dump(train_split, fp, sort_keys=True, indent=4)

    # Read transcripts
    transcripts_dir = os.path.join(conf.dataset_input_path, 'podcasts-transcripts')
    transcripts_paths = find_paths(dataframe, transcripts_dir, ".json")
    print("Number of transcripts found: ", len(transcripts_paths))
    transcripts_paths = sorted(transcripts_paths)
    # transcripts_paths = (transcripts_paths[:5]) # For now, only use 100 transcripts
    outer_folders, e_filenames, t_filenames = get_embed_transcript_paths(transcripts_paths)

    # Create output folders
    Path(os.path.join(conf.dataset_processed_path, "train")).mkdir(parents=True, exist_ok=True) 
    Path(os.path.join(conf.dataset_processed_path, "val")).mkdir(parents=True, exist_ok=True) 
    Path(os.path.join(conf.dataset_processed_path, "test")).mkdir(parents=True, exist_ok=True) 

    # Before processing all other data, first do the test set
    # That way we can move the evaluation task easier between devices.
    t = metadata_testset.episode_filename_prefix.tolist()
    process_first = []
    process_second = []
    for i in range(len(outer_folders)):
        fname = os.path.split(e_filenames[i])[-1].split('.')[0]
        if fname in t:
            process_first.append(i)
        else:
            process_second.append(i)

    process_idxs = process_first + process_second
    print(len(process_first))
    print(len(process_idxs))

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


for idx in process_idxs:
    # if idx > 2: 
    #     print("[del] break for now")
    #     break
#     elif idx < 40086:
#         last_output_folder = outer_folders[idx+1]
#         continue
    outer_folder = outer_folders[idx]
    filename = os.path.split(e_filenames[idx])[-1].split('.')[0]
    if outer_folder != last_output_folder:

        save_path_train = os.path.join(conf.dataset_processed_path, "train", last_output_folder.replace(os.sep, '_') + '_embeds.h5')
        save_path_val = os.path.join(conf.dataset_processed_path, "val", last_output_folder.replace(os.sep, '_') + '_embeds.h5')
        save_path_test = os.path.join(conf.dataset_processed_path, "test", last_output_folder.replace(os.sep, '_') + '_embeds.h5')
        print("saving {}/{}".format(idx, len(outer_folders)))

        # save_all(save_path, all_mean_embeddings, filenames, in_train_set, all_sentences, all_full_embeddings)
        if len(train_filenames) > 0:
            save_all(save_path_train, 
                    train_all_mean_embeddings, 
                    train_filenames, 
                    train_in_train_set, 
                    train_all_sentences, 
                    train_all_full_embeddings)
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
                    test_all_full_embeddings)

        last_output_folder = outer_folder
        train_all_sentences = []
        train_all_full_embeddings = []
        train_all_mean_embeddings = []
        train_filenames = []
        train_in_train_set = []
        val_all_sentences = []
        val_all_full_embeddings = []
        val_all_mean_embeddings = []
        val_filenames = []
        val_in_train_set = []
        test_all_sentences = []
        test_all_full_embeddings = []
        test_all_mean_embeddings = []
        test_filenames = []
        test_in_train_set = []

    if idx % 10 == 0:
        print("\t processing {}/{}".format(idx, len(outer_folders)))

    
    # Load yamnet embeddings and scores
    embed_path = os.path.join(conf.yamnet_embed_dir, outer_folder, e_filenames[idx])
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
        
        # Check if it belongs to train or test split
        train_filenames.extend([filename] * len(sentences))
        train_in_train_set.extend([train_split[filename]] * len(sentences))
        
        # add timestamps
        val_segment_tsps.extend(segment_ts)
        
    elif train_split[filename] == 'val':
        val_all_sentences.extend(sentences)
        val_all_mean_embeddings.extend(mean_embeddings)
        val_all_full_embeddings.extend(full_embeddings)

        # Check if it belongs to train or test split
        val_filenames.extend([filename] * len(sentences))
        val_in_train_set.extend([train_split[filename]] * len(sentences))
        
        # add timestamps
        val_segment_tsps.extend(segment_ts)

    elif train_split[filename] == 'test':
        test_all_sentences.extend(sentences)
        test_all_mean_embeddings.extend(mean_embeddings)
        test_all_full_embeddings.extend(full_embeddings)

        # Check if it belongs to train or test split
        test_filenames.extend([filename] * len(sentences))
        test_in_train_set.extend([train_split[filename]] * len(sentences))
        
        test_segment_tsps.extend(segment_ts)

    save_path_train = os.path.join(conf.dataset_processed_path, "train", last_output_folder.replace(os.sep, '_') + '_embeds.h5')
    save_path_val = os.path.join(conf.dataset_processed_path, "val", last_output_folder.replace(os.sep, '_') + '_embeds.h5')
    save_path_test = os.path.join(conf.dataset_processed_path, "test", last_output_folder.replace(os.sep, '_') + '_embeds.h5')
    print("saving {}/{}".format(idx, len(outer_folders)))

    # save_all(save_path, all_mean_embeddings, filenames, in_train_set, all_sentences, all_full_embeddings)
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

    exit(1)

    # Store train/test split on local disk
    train_split = dict(zip(subset.episode_filename_prefix, subset.train))
    train_split_filename = conf.sp_train_test_split
    with open(train_split_filename, 'w') as fp:
        json.dump(train_split, fp, sort_keys=True, indent=4)

    # Read transcripts
    transcripts_dir = os.path.join(conf.dataset_input_path, 'podcasts-transcripts')
    transcripts_paths = find_paths(subset, transcripts_dir, ".json")
    print("Number of transcripts found: ", len(transcripts_paths))
    transcripts_paths = sorted(transcripts_paths)
    # transcripts_paths = (transcripts_paths[:5000]) # For now, only use 100 transcripts
    outer_folders, e_filenames, t_filenames = get_embed_transcript_paths(transcripts_paths)

    # Create output folders
    Path(os.path.join(conf.dataset_processed_path, "train")).mkdir(parents=True, exist_ok=True) 
    Path(os.path.join(conf.dataset_processed_path, "test")).mkdir(parents=True, exist_ok=True) 


    last_output_folder = outer_folders[0]
    train_all_sentences = []
    train_all_full_embeddings = []
    train_all_mean_embeddings = []
    train_filenames = []
    train_in_train_set = []
    test_all_sentences = []
    test_all_full_embeddings = []
    test_all_mean_embeddings = []
    test_filenames = []
    test_in_train_set = []

    for idx, outer_folder in enumerate(outer_folders):
        if idx < 40086:
            last_output_folder = outer_folders[idx+1]
            continue
        filename = os.path.split(e_filenames[idx])[-1].split('.')[0]
        if outer_folder != last_output_folder:
            
            save_path_train = os.path.join(conf.dataset_processed_path, "train", last_output_folder.replace(os.sep, '_') + '_embeds.h5')
            save_path_test = os.path.join(conf.dataset_processed_path, "test", last_output_folder.replace(os.sep, '_') + '_embeds.h5')
            print("saving {}/{}".format(idx, len(outer_folders)))

            # save_all(save_path, all_mean_embeddings, filenames, in_train_set, all_sentences, all_full_embeddings)
            if len(train_filenames) > 0:
                save_all(save_path_train, 
                        train_all_mean_embeddings, 
                        train_filenames, 
                        train_in_train_set, 
                        train_all_sentences, 
                        train_all_full_embeddings)
            if len(test_filenames) > 0:
                save_all(save_path_test, 
                        test_all_mean_embeddings, 
                        test_filenames, 
                        test_in_train_set, 
                        test_all_sentences, 
                        test_all_full_embeddings)
  
            last_output_folder = outer_folder
            train_all_sentences = []
            train_all_full_embeddings = []
            train_all_mean_embeddings = []
            train_filenames = []
            train_in_train_set = []
            test_all_sentences = []
            test_all_full_embeddings = []
            test_all_mean_embeddings = []
            test_filenames = []
            test_in_train_set = []

        if idx % 10 == 0:
            print("\t processing {}/{}".format(idx, len(outer_folders)))

        # Load yamnet embeddings and scores
        embed_path = os.path.join(conf.yamnet_embed_dir, outer_folder, e_filenames[idx])
        yamnet_embedding = pd.read_hdf(embed_path)
        
        # Load the transcript
        transcript_path = os.path.join(transcripts_dir, outer_folder, t_filenames[idx])
        transcript_json = load_transcript(transcript_path)
        
        # Extract all sentences and corresponding words
        sentences, timestamps, mean_embeddings, full_embeddings = extract_transcript(transcript_json, yamnet_embedding)
        if train_split[filename]:
            train_all_sentences.extend(sentences)
            train_all_mean_embeddings.extend(mean_embeddings)
            train_all_full_embeddings.extend(full_embeddings)
            # Check if it belongs to train or test split
            train_filenames.extend([filename] * len(sentences))
            train_in_train_set.extend([train_split[filename]] * len(sentences))

        else:
            test_all_sentences.extend(sentences)
            test_all_mean_embeddings.extend(mean_embeddings)
            test_all_full_embeddings.extend(full_embeddings)
        
            # Check if it belongs to train or test split
            test_filenames.extend([filename] * len(sentences))
            test_in_train_set.extend([train_split[filename]] * len(sentences))

    save_path_train = os.path.join(conf.dataset_processed_path, "train", last_output_folder.replace(os.sep, '_') + '_embeds.h5')
    save_path_test = os.path.join(conf.dataset_processed_path, "test", last_output_folder.replace(os.sep, '_') + '_embeds.h5')
    print("saving {}/{}".format(idx, len(outer_folders)))

    # save_all(save_path, all_mean_embeddings, filenames, in_train_set, all_sentences, all_full_embeddings)
    if len(train_filenames) > 0:
        save_all(save_path_train, 
                train_all_mean_embeddings, 
                train_filenames, 
                train_in_train_set, 
                train_all_sentences, 
                train_all_full_embeddings)
    if len(test_filenames) > 0:
        save_all(save_path_test, 
                test_all_mean_embeddings, 
                test_filenames, 
                test_in_train_set, 
                test_all_sentences, 
                test_all_full_embeddings)





    
    
    


