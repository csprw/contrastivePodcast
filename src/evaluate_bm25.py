"""
Contrastive multimodal learning: Evaluation
Author: Casper Wortmann
Usage: python evaluate.py
"""
# Load all modules
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
from omegaconf import OmegaConf
from dacite import from_dict
from argparse import ArgumentParser
import torch 
from rank_bm25 import BM25Okapi

conf = OmegaConf.load("./config.yaml")

from evaluate_topic import Evaluator, get_sent_audio
from train import mmModule, Cfg
from train import MMloader
from utils import preprocess, read_metadata_subset


def bm25_evaluation(evaluator):
    corpus = preprocess(evaluator.all_sents)
    bm25 = BM25Okapi(corpus)
    k=5000

    targets = evaluator.all_targs
    relevant_eps = evaluator.relevant_eps
    results_bm25 = {}

    overall_rank = []
    mrr = []
    for query_idx in range(len(evaluator.queries)):
        cur_query = evaluator.queries[query_idx]
        print("---This is query: ", cur_query)

        query = preprocess([cur_query])[0]
        doc_scores = bm25.get_scores(query)
        top_n_idx = np.argsort(doc_scores)[-k:][::-1]
        
        indices = top_n_idx
        predicted_segs = list(dict.fromkeys([targets[i] for i in indices]))
        predicted_epis = list(dict.fromkeys([i.split("_")[0] for i in predicted_segs]))

        correct_episodes = relevant_eps[query_idx]
        ranks = []
        for possible_target in correct_episodes:
            if possible_target in predicted_epis:
                estimated_position = predicted_epis.index(possible_target)
                ranks.append( estimated_position)
        if len(ranks) > 0:
            best_estimation = np.min(ranks)
        else:
            best_estimation = k
            
        overall_rank.append(best_estimation)
        mrr.append(1 / (best_estimation + 1))

    results_bm25['names'] = 'bm25'
    results_bm25['rank']= np.mean(overall_rank)
    results_bm25['rank_var'] = np.var(overall_rank)
    results_bm25['rank_std']= np.std(overall_rank)
    results_bm25['mrr']= np.mean(mrr)
    results_bm25['mrr_var']= np.std(mrr)
    results_bm25['mrr_std']= np.var(mrr)
    return results_bm25

def main(args):
    print("[main] Evaluate")

    metadata_testset, topics_df, topics_df_targets = read_metadata_subset(conf, traintest='test')
    print(len(metadata_testset))

    # Read topic metadata from dict
    sent_query_output_path = os.path.join(conf.dataset_path, 'TREC_topic', 'topic_dict_query.json')
    sent_descr_output_path = os.path.join(conf.dataset_path, 'TREC_topic', 'topic_dict_descr.json')

    ## Read dictionary with summaries
    with open(sent_query_output_path, 'r') as f:
        sent_query_dict = json.load(f)
    with open(sent_descr_output_path, 'r') as f:
        sent_descr_dict = json.load(f)
    print(sent_descr_dict.keys())
    sent_descr_audio_dict = get_sent_audio(conf.sent_topic_descr_embed_dir)
    sent_query_audio_dict = get_sent_audio(conf.sent_topic_query_embed_dir)

    # Add episode score
    val_df = topics_df_targets.copy()
    positive_episodes = val_df.loc[val_df['bin_score'] == 1].copy()
    positive_eplist = positive_episodes['episode_uri'].tolist()
    for i, row in val_df.iterrows():
        ifor_val = 0
        if row['episode_uri'] in positive_eplist:
            ifor_val = 1
        val_df.loc[i,'ep_score'] = ifor_val
    val_df.ep_score = val_df.ep_score.astype(int)

    # Loading the model.
    model_path = Path(args.model_weights_path).parents[1]
    model_weights_path = args.model_weights_path
    model_config_path = os.path.join(model_path, 'config.json')

    # Opening JSON file
    f = open(model_config_path)
    model_config = json.load(f)
    CFG = from_dict(data_class=Cfg, data=model_config)
    CFG.sp_path =  conf.sp_path
    CFG.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataloader of test set
    data_loader = MMloader(CFG)

    # Load the model
    full_model = mmModule(CFG)
    full_model.load_state_dict(torch.load(model_weights_path,  map_location=CFG.device)) 

    full_model = full_model.to(CFG.device)     
    full_model.eval()

    # Calculate embeddings for test set
    evaluator = Evaluator(CFG, model_path, full_model, data_loader, 
            save_intermediate=False, calc_acc = True)
    max_samples = evaluator.get_max_data()

    # print(max_samples)
    # max_samples = 128* 4

    evaluator.encode_testset_new(max_samples) 
    evaluator.encode_sent_descr(topics_df, field='query', sent_dict=sent_query_dict, audio_dict=sent_query_audio_dict)
    evaluator.encode_sent_descr(topics_df, field='description', sent_dict=sent_descr_dict, audio_dict=sent_descr_audio_dict)
    evaluator.add_query_labels(topics_df, val_df)
    evaluator.queries = topics_df['query'].tolist()

    results = bm25_evaluation(evaluator)

    json_out = str(Path(args.model_weights_path).parents[1])
    with open(os.path.join(json_out, 'topic_bm25_results.json'), 'w') as fp:
        json.dump(results, fp, indent=4)



if __name__ == "__main__":
    print("[evaluate] reading data from: ")
    print(conf.dataset_path)

    # Parse flags in command line arguments
    parser = ArgumentParser()

    parser.add_argument('--model_weights_path', type=str, default="logs/random_model/output/full_model_weights.pt",
                        help='Folder where model weights are saved.')
    args, unparsed = parser.parse_known_args()

    main(args)
