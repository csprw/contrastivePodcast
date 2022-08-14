"""
Self-supervised Contrastive Learning from Podcast Audio and Transcripts.
"""
import json, csv
import os
import math
import pandas as pd
import numpy as np
import pprint
import torch 
from matplotlib import pyplot as plt
import seaborn as sns
from nltk import tokenize
import logging
from datetime import datetime
from dacite import from_dict

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

def get_log_name(args):
    """
    Returns name of the current run.
    """
    log_name = "{}m-{}_{}".format(int(args.max_train_samples / 1000000), 
            args.audio_proj_head, 
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_name = os.path.join(args.log_dir, log_name)
    return log_name

def setup_config(args, dc, device='cpu'):
    """
    Creates a configuration file and saves it to disk. 
    Inputs:
        dc: Dataclass configuration of the training run. 
    Outputs:
        fullcfg: Dataclass configuration with argparse objects merged into it. 
    """
    # Set the seed and create output directories
    set_seed(args)
    log_name = get_log_name(args)
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

    fullcfg = from_dict(data_class=dc, data=config)

    # Save config.
    config_dir = os.path.join(log_name, 'config.json')
    with open(config_dir, "w") as file:
        json.dump(config, file, indent=4, sort_keys=True)
    return fullcfg

def print_best_results(eval_csv_filename, dur, out_dir):
    """ 
    Prints the best results of the training run. 
    Inputs:
        eval_csv_filename: path to the evaluation metrics of the run. 
        dur: duration of the run (seconds). 
        out_dir: path to the folder where best results should be saves. 
    """
    csv_file = pd.read_csv(eval_csv_filename)
    best_idx = csv_file['mean_acc'].idxmax()
    best_metrics = {'duration': dur}
    for k, v in csv_file.iloc[best_idx].to_dict().items():
        best_metrics[k] = v

    outfile = os.path.join(out_dir, 'best_results.json')
    with open(outfile, "w") as file:
        json.dump(best_metrics, file, indent=4, sort_keys=True)
    print("---- Best epoch results ----")
    pprint(best_metrics)

def get_metrics(audio_logits, text_logits, ground_truth):
    """
    Adds accuracy metrics to the log file. 
    Inputs:
        audio_logits: output of the audio projection head.
        text_logits: output of text projection head.
        ground_truth: the targets (a diagonal matrix).
    Outputs:
        metrics: dict containing metrics of the prediction task. 
    """
    metrics = {}
    accs = []
    logits = {"audio": audio_logits, "text": text_logits}
    
    for name, logit in logits.items():
        acc = torch.mean((logit.argmax(dim=-1) == ground_truth).float()).item()
        accs.append(acc)
        metrics[f"{name}_acc"] = acc
    metrics['mean_acc'] = np.mean(accs)
    return metrics

def to_plot(filename, column='accuracy', title="Val accuracy"):
    """
    Creates a plot of a training curve. 
    """
    csv_file = pd.read_csv(filename)

    ax = sns.lineplot(x=csv_file.steps, y=csv_file[column])
    ax.set(title=title)
    
    output_dir = os.path.split(filename)[0]
    output_title = os.path.split(filename)[-1].split('.')[0]
    output_name = os.path.join(output_dir, output_title + "_" + column + '.jpg')
    plt.savefig(output_name)
    plt.close()


def read_metadata_subset(conf, traintest='test'):
    """
    Read the subset of metadata which we will use for the topic task
    Inputs:
        traintest: read the train or test set of the topic task. 
    Outputs:
        metadata_subset: metadata after preprocessing (removed duplicates etc).
        topics_df: dataframe containing topic-test set queries and descriptions.
        topics_df_targets: df containing the relevance assessments of the episodes
            in the topic test set. 
    """
    metadata_path = os.path.join(conf.dataset_path, "metadata.tsv")
    metadata  = pd.read_csv(metadata_path, delimiter="\t")
    print("[utils] metadata loaded ", len(metadata))

    target_dir = os.path.join(conf.dataset_path, 'TREC_topic')
    colnames = ['num', 'unk', 'episode_uri_time', 'score']

    if traintest == 'test':
        # Read target data
        topics_df_path = os.path.join(target_dir, 'podcasts_2020_topics_test.xml')
        topics_df = pd.read_xml(topics_df_path)

        # Read annotations (line deliminter is different for train and test)
        topics_df_targets_path = os.path.join(target_dir, '2020_test_qrels.list.txt')
        topics_df_targets = pd.read_csv(topics_df_targets_path, delimiter=r"\s+", lineterminator='\n', names=colnames)
        print("[utils] topics_df_targets loaded for test set", len(topics_df_targets))

    if traintest == 'train':
        # Read target data
        topics_df_path = os.path.join(target_dir, 'podcasts_2020_topics_test.xml')
        topics_df = pd.read_xml(topics_df_path)

        # Read annotations (line deliminter is different for train and test)
        topics_df_targets_path = os.path.join(target_dir, '2020_train_qrels.list.txt')
        topics_df_targets = pd.read_csv(topics_df_targets_path, sep='\t', lineterminator='\n', names=colnames)
        print("[utils] topics_df_targets loaded for train set", len(topics_df_targets))

    # Remove float '.0' from column string
    topics_df_targets['episode_uri_time'] = topics_df_targets['episode_uri_time'].str.replace(r'.0$', '', regex=True)

    # Add a binary score
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
    print("[utils] metadata_subset loaded ", len(metadata_subset))

    # remove 'spotify:episode:' from column so we can match items
    topics_df_targets['episode_uri_time'] = topics_df_targets.episode_uri_time.str.replace(r'spotify:episode:', '')
    topics_df_targets['episode_uri'] = topics_df_targets.episode_uri_time.str.replace(r'spotify:episode:', '')

    # Remove timestamps from episode uri
    topics_df_targets['episode_uri'] = topics_df_targets['episode_uri'].str.split('_').str[0]

    # TODO: DELETE THESE TWO LINES
    metadata_subset = metadata_subset.sort_values('score')
    return metadata_subset, topics_df, topics_df_targets



def get_embed_transcript_paths(transcripts_paths):
    """
    Returns the path of all embeddings (.h5 files) in a directory. 
    """
    outer_folders = []
    e_filenames = []
    t_filenames = []
    for p in transcripts_paths:
        splitted_p = os.path.normpath(p).split(os.path.sep)
        outer_folders.append(os.path.join(splitted_p[-4], splitted_p[-3]))
        t_filenames.append(os.path.join(splitted_p[-2], splitted_p[-1]))
        e_filenames.append(os.path.join( splitted_p[-2], splitted_p[-1].split('.')[-2]+".h5"))

    return outer_folders, e_filenames, t_filenames

def get_sent_indexes(sentences):
    """ Returns a list of tuples, with (starttime, endtime) for each sentence. """
    indexes = []
    lb = 0
    ub = 0
    for s in sentences:
        extra_indexes = len(s.split())
        ub += extra_indexes 
        indexes.append((lb, ub-1))
        lb += extra_indexes
    return indexes

def randomize_model(model):
    """ Loads a model with random weights. """
    for module_ in model.named_modules(): 
        if isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
            module_[1].weight.data.normal_(mean=0.0, std=1.0)
        elif isinstance(module_[1], torch.nn.LayerNorm):
            module_[1].bias.data.zero_()
            module_[1].weight.data.fill_(1.0)
        if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
            module_[1].bias.data.zero_()
    return model

def preprocess(text):
    """ Preprocesses a text-corpus. """
    tokenized_corpus = [doc.split(" ") for doc in text]
    processed = [[j.lower() for j in i] for i in tokenized_corpus]
    return processed


def load_transcript(path):
    """
    Loads a transcript from the SP dataset.
    Input: 
        path: path to a transcript .json file
    output:
        transcript: the corresponding transcript as dicitonary. 
    """
    with open(path, "r") as file:
        transcript = json.load(file)
    return transcript
    
def extract_transcript(transcript_json, yamnet_embedding):
    '''
    Extracts sentences and corresponding timestamps from a transcript, 
        and returns the corresopnding yamnet embeddings. 
    Input: 
        transcript_json: Json file containing a podcast transcript
        yamnet_embedding: The precomputed yamnet embedding. 
    Output:
        sentences: a list of sentences
        timestamps: a list of tuples [(starttime, endtime)]
        segment_timestamps: the timestamps per segment
        mean_embeddings: the mean of the yamnet embeddings inbetween (starttime, endtime)
        full_embeddings: all yamnet embeddings inbetween (starttime, endtime)
    '''
    sentences = []
    timestamps = []
    segment_timestamps = []
    mean_embeddings = []
    full_embeddings = []
    
    seg_duration = 60.0
    
    for result in transcript_json['results'][:-1]:
        if len(result['alternatives'][0]) == 0:
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
    Returns the paths to the .h5py files corresponding to the metadata df. 
    Finds the filepaths in base_folder with extension file_extension.
    """
    paths = []
    for i in range(len(metadata)):
        relative_path = os.path.join(
            metadata.show_filename_prefix.iloc[i][5].upper(), 
            metadata.show_filename_prefix.iloc[i][6].upper(), 
            metadata.show_filename_prefix.iloc[i],
            metadata.episode_filename_prefix.iloc[i])
        paths.append(os.path.join(base_folder, relative_path + file_extension))
    return paths

