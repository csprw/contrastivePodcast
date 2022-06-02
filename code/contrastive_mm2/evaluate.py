"""
Contrastive multimodal learning: Evaluation
Author: Casper Wortmann
Usage: python evaluate.py
"""
import h5py
import json
import os
from pathlib import Path
import math
import pandas as pd
import numpy as np
import sys
from collections import defaultdict

from argparse import ArgumentParser
from dacite import from_dict
from omegaconf import OmegaConf
from torch import topk
from sklearn.metrics import ndcg_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

import torch 
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence


sys.path.append('../scripts/') 
from prepare_index_sentencelevel2 import read_metadata_subset
from gru_test2 import mmModule, Cfg
from gru_test2 import MMloader
from src.data import load_metadata, find_paths, relative_file_path
conf = OmegaConf.load("./config.yaml")


def precision_at_k(predicted, k):
    pred = (predicted[:k])
    relevant = np.sum(pred)
    num_recommended = float(k)
    p_at_k = relevant / num_recommended
    return p_at_k

# Recall@k = (# of recommended items @k that are relevant) / (total # of relevant items)
def recall_at_k(predicted, num_relevant, k):
    if num_relevant == 0:
        return None
    pred = (predicted[:k])
    relevant = np.sum(pred)
    r_at_k = relevant / num_relevant

    return r_at_k



class Evaluator(object):
    def __init__(self, CFG, full_model, test_loader, tokenizer, model_path):
        """

        """
        self.model_path = model_path
        self.model = full_model
        self.test_loader = test_loader
        self.device = CFG.device
        self.scale = CFG.scale
        self.bs = CFG.batch_size
        self.embed_dim = CFG.mutual_embedding_dim
        self.tokenizer = tokenizer

    @torch.no_grad()
    def encode_data(self):
        print("Encode data")
        accs = []
        encoded_targets = []
        max_steps = len(self.test_loader) 


        sent_encoding = np.zeros((max_steps * self.bs, self.embed_dim)) 
        audio_encoding = np.zeros((max_steps * self.bs, self.embed_dim))

        # This is for now to code faster, delete it
        # my_tmp_file = Path("tmp_validation.hdf5")
        # if my_tmp_file.is_file():
        #     f1 = h5py.File("tmp_validation.hdf5", "r")
        #     read_something = True
        # else:
        #     f1 = h5py.File("tmp_validation.hdf5", "w")
        #     read_something= False
        #  # This is for now to code faster, delete it
        # my_tmp_file = Path("tmp_validation2.hdf5")
        # if my_tmp_file.is_file():
        #     f2 = h5py.File("tmp_validation2.hdf5", "r")
        #     read_something = True
        # else:
        #     f2 = h5py.File("tmp_validation2.hdf5", "w")
        
        for step, (tok_sentences, audio_features, seq_len, targets, full_text) in enumerate(self.test_loader):
            # if step > 2:
            #     print("[del] Break!")
            #     break

            # all_targs_del = '-'.join(targets)
            all_targs_del = targets[0]

            my_tmp_file = os.path.join("tmp_v2", all_targs_del+".hdf5")
            if Path(my_tmp_file).is_file():
                print("load it from file")
                f1 = h5py.File(my_tmp_file, "r")
                reps_sentences = torch.tensor(f1['reps_sentences'])
                reps_audio = torch.tensor(f1['reps_audio'])
                f1.close()

            else:
                print("create it: ", step, len(self.test_loader))
                # print("This is ful_text: ", full_text)
                f1 = h5py.File(my_tmp_file, "w")
                tok_sentences = tok_sentences.to(self.device)
                audio_features = audio_features.to(self.device)  

                reps_sentences = self.model.text_model(tok_sentences)['sentence_embedding']
                reps_audio = self.model.audio_model((audio_features, seq_len))

                reps_audio = reps_audio / reps_audio.norm(dim=1, keepdim=True)
                reps_sentences = reps_sentences / reps_sentences.norm(dim=1, keepdim=True)

                f1.create_dataset('reps_sentences', data=reps_sentences.cpu())
                f1.create_dataset('reps_audio', data=reps_audio.cpu())
                f1.create_dataset('targets', data=np.array(targets, dtype=h5py.special_dtype(vlen=str)))
                f1.create_dataset('full_text', data=np.array(full_text, dtype=h5py.special_dtype(vlen=str)))
                f1.close()

            # Add the encodings to a matrix
            idxs = np.arange(step * self.bs, step*self.bs + self.bs)
            sent_encoding[idxs, :] = reps_sentences.cpu().numpy().copy()
            audio_encoding[idxs, :] = reps_audio.cpu().numpy().copy()
            encoded_targets.extend(targets)

            # Calculate accuracy 
            del_skip_acc = True
            if del_skip_acc:
                accs = [0,0]
            else:
                audio_logits =  (reps_audio @ reps_sentences.t()) * self.scale
                text_logits = audio_logits.t()
                
                audio_probs = audio_logits.softmax(dim=-1).cpu().numpy()
                text_probs = text_logits.softmax(dim=-1).cpu().numpy()
                ground_truth = torch.arange(self.bs)

                audio_acc = torch.eq(torch.tensor(audio_probs.argmax(axis=0)), ground_truth).sum() / ground_truth.shape[0]
                text_acc = torch.eq(torch.tensor(text_probs.argmax(axis=0)), ground_truth).sum() / ground_truth.shape[0]
                accs.append((audio_acc.item() + text_acc.item()) / 2)


        mean_acc = np.mean(accs)
        print("[evaluate] acc: ", mean_acc)
        self.text_encoding = sent_encoding
        self.audio_encoding = audio_encoding
        self.encoded_targets = encoded_targets
        return mean_acc

    def get_query_data(self, topics_df):
        print("[del] get query data")
        print("Tokenizing queries")
        query_field =  'query' # TODO: ook description?
        queries = topics_df[query_field].tolist()

        self.tokenized_queries = self.tokenizer(
            queries, padding=True, truncation=True, max_length=32, return_tensors='pt', return_token_type_ids=True,
        ).to(self.device)


        print("[del] Creating padding of yamnets of queries")
        query_yamnets = []
        query_lengths = []
        for idx, row in topics_df.iterrows():
            query_num = row.num
            query_embed_path  = os.path.join(conf.yamnet_query_embed_path, str(query_num) + ".h5")
            inbetween_yamnet_embeds = pd.read_hdf(query_embed_path)

            query_lengths.append(len(inbetween_yamnet_embeds))
            tmp = torch.Tensor(inbetween_yamnet_embeds.values)

            query_yamnets.append(tmp)
            #lengths.append(len(example[1]))
        self.query_lengths = query_lengths
        self.padded_query_yamnets = pad_sequence(query_yamnets, batch_first=True).to(self.device)

    @torch.no_grad()
    def encode_query_data(self):
        print("Creating query embeddings now:")
        query_text_repr = []
        query_audio_repr = []
        query_field = 'query'

        with torch.no_grad():
            # print("[del] get embeds: ")
            reps_sentences = self.model.text_model(self.tokenized_queries)['sentence_embedding']
            reps_audio = self.model.audio_model((self.padded_query_yamnets, self.query_lengths))

            query_norm_reps_audio = reps_audio / reps_audio.norm(dim=1, keepdim=True)
            query_norm_reps_text = reps_sentences / reps_sentences.norm(dim=1, keepdim=True)

            query_text_repr.append(query_norm_reps_text)
            query_audio_repr.append(query_norm_reps_audio)
    
        # query_audio_repr = torch.cat(query_audio_repr, dim=0).cpu()
        # query_text_repr = torch.cat(query_text_repr, dim=0).cpu()

        self.query_audio_encoding = torch.cat(query_audio_repr, dim=0).cpu()
        self.query_text_encoding = torch.cat(query_text_repr, dim=0).cpu()


    def topic_task(self, val_df, topics_df):
        print("[del] in topic task")

        # Validate for all rows for which we have encoded data.
        ids_we_can_use = [m.split('_')[0] for m in self.encoded_targets]
        print("[del] we can use {} ids".format(len(set(ids_we_can_use))))
        print("[del] val_df before: ", len(val_df))
        val_df = val_df[val_df['episode_uri'].isin(ids_we_can_use)]
        print("[del] val_df after (should be same in the end!!): ", len(val_df))

        k = 100
        results = {}
        

        combis = [[self.query_text_encoding, self.text_encoding], [self.query_audio_encoding, self.audio_encoding], [self.query_text_encoding, self.audio_encoding]]
        names = ["text2text", "audio2audio", "text2audio"]

        for idx, combi in enumerate(combis):
            name = names[idx]
            results[name] = defaultdict(list)
            similarity = 100 * combi[0] @ combi[1].T

            # TODO: also description?
            query_field = 'query'
            probs = similarity.softmax(dim=-1).cpu() # moet over zinnen zijn

            top_probs, top_labels = probs.topk(k, dim=-1)

            for row_idx in range(len(topics_df)):
                query = topics_df[query_field][row_idx]
                print("query: ", query)
                pred_ind = top_labels[row_idx].tolist()

                pred_epis = [self.encoded_targets[val].split('_')[0] for val in pred_ind]
                query_num = topics_df['num'][row_idx]
                print("[del] checking for {} and num {} ".format(row_idx, query_num))
                
                tmp = val_df[(val_df.num == query_num) & (val_df.ep_score==1)]
                tmp = tmp['episode_uri'].tolist()
                num_episodes_relevant = len(set(tmp))
                print("[del] Relevant episodes for Query: ", num_episodes_relevant)
                        
                ep_scores = []
                for episode_uri in pred_epis:
                    ep_score_row = val_df.loc[val_df['episode_uri'] == episode_uri]
                    if len(ep_score_row) > 0 and ep_score_row.num.values[0] == query_num:
                        ep_scores.append(1)
                    else:
                        ep_scores.append(0)
                
                # pred_episodes[query] = {}
                # # pred_episodes[query]['episodes'] = [matrix_targets[val].split('_')[0] for val in pred_ind]
                # pred_episodes[query]['ep_score'] = ep_scores
                
                # pred_episodes[query]['prec@3'] = precision_at_k(ep_scores, 3)
                # pred_episodes[query]['prec@10'] = precision_at_k(ep_scores, 10)
                # pred_episodes[query]['prec@30'] = precision_at_k(ep_scores, 30)

                targets = [[1] * num_episodes_relevant + [0] * (len(ep_scores) - num_episodes_relevant)]
                targets = [targets[0][:k]]
                print("Targets: ", targets)
                print("scores: ", ep_scores)
                
                ndcg_ep_score = ndcg_score(targets, [ep_scores], k=30)
                # pred_episodes[query]['ndcg'] = ndcg_ep_score

                # ap = average_precision_score(targets[0], ep_scores)
                # print("ROC for: ", targets[0], ep_scores)
                if  not all(np.array(targets[0]) == 0):
                    
                    if all(np.array(ep_scores) ==0):
                        roc = 0.0
                        results[name]['roc'].append(roc)
                    else:
                        roc = roc_auc_score(targets[0], ep_scores)
                        results[name]['roc'].append(roc)
                
                results[name]['prec@3'].append(precision_at_k(ep_scores, 3))
                results[name]['prec@10'].append(precision_at_k(ep_scores, 10))
                results[name]['prec@30'].append(precision_at_k(ep_scores, 30))
                results[name]['prec@100'].append(precision_at_k(ep_scores, 100))
                results[name]['ndcg'].append(ndcg_ep_score)
                
                print("done query {}, p@10 {}, ndcg: {}".format(query_num, precision_at_k(ep_scores, 10), ndcg_ep_score))

        return results

    def save_results(self, results):
        mean_results = {}
        for name, metric in results.items():
            mean_results[name] = {}
            for metric_name, vals in metric.items():
                mean_results[name][metric_name] = np.mean(vals)
                
        print(mean_results)
        out = os.path.join(self.model_path, "topic_results.json")
        with open(out, "w") as f:
            json.dump(mean_results, f, indent=4)


def main(args):
    print("[del] in main")
    # Reading the metadata.
    metadata_testset, topics_df, topics_df_targets = read_metadata_subset(conf, traintest='test')
    print(len(metadata_testset))

    # Remove duplicate rows
    metadata_testset = metadata_testset.drop_duplicates(subset=['episode_filename_prefix']).sort_values('episode_filename_prefix')
    print("[main] Topic test set loaded: ", len(metadata_testset))

    # TODO: move to seperate function
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
    model_path = args.model_path
    print("[Load model] from ", model_path)
    model_weights_path = os.path.join(model_path, "output/full_model_weights.pth")
    model_config_path = os.path.join(model_path, 'config.json')

    # Opening JSON file
    f = open(model_config_path)
    model_config = json.load(f)
    print("using config: , ", model_config)
    CFG = from_dict(data_class=Cfg, data=model_config)
    CFG.device = "cuda" if torch.cuda.is_available() else "cpu"
    # mutual_embed_dim = CFG.final_projection_dim
    print("[Load model] config loaded: ", CFG)

    # Load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(CFG.text_model_name)

    # Create dataloader of test set
    data_loader = MMloader(CFG)
    test_loader = data_loader.test_loader 

    # Load the model
    full_model = mmModule(CFG)
    full_model.load_state_dict(torch.load(model_weights_path,  map_location=CFG.device))              
    full_model = full_model.to(CFG.device)     
    full_model.eval()

    tokenizer = data_loader.test_dataset.tokenizer
    evaluator = Evaluator(CFG, full_model, test_loader, tokenizer, model_path)
    
    acc = evaluator.encode_data()
    print("[Test accuracy]: ", acc)
    
    evaluator.get_query_data(topics_df)
    evaluator.encode_query_data()
    results = evaluator.topic_task(val_df, topics_df)
    evaluator.save_results(results)
    # sent_encoding, audio_encoding, targets, acc = evaluator.encode_data()
    

if __name__ == "__main__":
    print("[evaluate] reading data from: ")
    print(conf.dataset_input_path)

    # Parse flags in command line arguments
    parser = ArgumentParser()

    parser.add_argument('--model_path', type=str, default="logs/windows_gru2-clip_loss_followup",
                        help='Folder where model weights are saved.')
    args, unparsed = parser.parse_known_args()

    main(args)
