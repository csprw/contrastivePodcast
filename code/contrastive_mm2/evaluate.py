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
from train import mmModule, Cfg
from train import MMloader
from src.data import load_metadata, find_paths, relative_file_path
conf = OmegaConf.load("./config.yaml")


def prec_at_k(scores, k):
    #Precision@k = (# of recommended items @k that are relevant) / (# of recommended items @k)
    scores_k = scores[:k]

    prec = np.sum(scores_k) / len(scores_k)
    return prec

def rec_at_k(scores, num_rel, k):
    # (of recommended items that are relevant @k)/(total # of relevant items)
    scores_k = scores[:k]
    rec = np.sum(scores_k) / num_rel
    return rec
        

# def text_to_embed(tokenizer, text, full_model):
#     tokenized_text = tokenizer(
#         text, padding=True, truncation=True, max_length=32, return_tensors='pt', return_token_type_ids=True,
#     )
    
#     with torch.no_grad():
#         reps_sentences = full_model.text_model(tokenized_text)['sentence_embedding']
#         embed = reps_sentences / reps_sentences.norm(dim=1, keepdim=True)

#     return embed


# def audio_to_embed(CFG, yamnets, query_lengths, full_model):
#     if CFG.audio_proj_head == 'gru':
#         padded_yamnets = pad_sequence(yamnets, batch_first=True)

#         with torch.no_grad():
#             reps_audio = full_model.audio_model((padded_yamnets, query_lengths))
#             embed = reps_audio / reps_audio.norm(dim=1, keepdim=True)
#     else:
#         raise NotImplementedError
#     return embed


class Evaluator(object):
    def __init__(self, CFG, model_path, full_model, data_loader, 
        save_intermediate, calc_acc):
        """
        Evaluator object
        """
        self.model_path = model_path
        self.model = full_model
        self.device = CFG.device
        self.scale = CFG.scale
        self.bs = CFG.batch_size
        self.embed_dim = CFG.mutual_embedding_dim
        self.tokenizer =  data_loader.test_dataset.tokenizer
        self.test_loader = data_loader.test_loader

        self.audio_proj_head = CFG.audio_proj_head
        self.fixed_scale = CFG.scale
        
        self.test_dir = os.path.join(conf.sp_path, 'test')
        
        self.save_intermediate = save_intermediate
        self.calc_acc = calc_acc
        self.acc = 0.0

        # self.topic_output =  os.path.join(conf.yamnet_embed_path, "topic_embeddings")   # TODO: change to CFG
        self.topic_output =  os.path.join(conf.topic_embed_path, "topic_embeddings", os.path.split(model_path)[-1]) 
        Path(self.topic_output).mkdir(parents=True, exist_ok=True)

        
    def get_max_data(self):
        "returns max number of sentences to encode"
        h5py_files = list(Path(self.test_dir).glob('*.h5'))
        max_samples = 0
        for h5idx, h5py_file in enumerate(h5py_files):    
            f = h5py.File(h5py_file, 'r')
            max_samples += len(f['sentences'])
        print("same? ", max_samples)
        max_samples =  len(self.test_loader) * self.bs
        print("same? ", max_samples)
    
        return max_samples

    def audio_to_embed(self, yamnets, query_lengths):
        if self.audio_proj_head == 'gru':
            padded_yamnets = pad_sequence(yamnets, batch_first=True).to(self.device)

            with torch.no_grad():
                reps_audio = self.model.audio_model((padded_yamnets, query_lengths))
                embed = reps_audio / reps_audio.norm(dim=1, keepdim=True)
        else:
            raise NotImplementedError
        return embed.cpu()

    def text_to_embed(self, text):
        tokenized_text = self.tokenizer(
            text, padding=True, truncation=True, max_length=32, return_tensors='pt', return_token_type_ids=True,
        )
        
        
        with torch.no_grad():
            tokenized_text = tokenized_text.to(self.device)
            reps_sentences = self.model.text_model(tokenized_text)['sentence_embedding']
            embed = reps_sentences / reps_sentences.norm(dim=1, keepdim=True)

        return embed.cpu()

    @torch.no_grad()  
    def encode_testset_new(self, max_samples):
        accs = []
        text_encoding = np.zeros((max_samples, self.embed_dim)) 
        audio_encoding = np.zeros((max_samples, self.embed_dim)) 
        
        all_sents = np.zeros(max_samples, dtype=object)
        all_targs = np.zeros(max_samples, dtype=object)
        
        for step, (tok_sentences, audio_features, seq_len, targets, sents) in enumerate(self.test_loader):
            
            #my_tmp_file = os.path.join("tmpv5", targets[0] +'_'+ str(step)+".hdf5") # for new datasets change this!
            my_tmp_file = os.path.join(self.topic_output, str(step)+".hdf5")
            
            if Path(my_tmp_file).is_file():
                print("load it from file: ", step, len(self.test_loader))
                f1 = h5py.File(my_tmp_file, "r")
                text_batch = torch.tensor(np.array(f1['text_batch']))
                audio_batch = torch.tensor(np.array(f1['audio_batch']))
                f1.close()
                
            else:
                print("Calculating: ", step, len(self.test_loader), my_tmp_file)
                tok_sentences = tok_sentences.to(self.device)
                audio_features = audio_features.to(self.device)  

                reps_sentences = self.model.text_model(tok_sentences)['sentence_embedding']
                reps_audio = self.model.audio_model((audio_features, seq_len))

                audio_batch = reps_audio / reps_audio.norm(dim=1, keepdim=True)
                text_batch = reps_sentences / reps_sentences.norm(dim=1, keepdim=True)

                if self.save_intermediate:
                    f1 = h5py.File(my_tmp_file, "w")
                    f1.create_dataset('text_batch', data=text_batch.cpu())
                    f1.create_dataset('audio_batch', data=audio_batch.cpu())
                    f1.create_dataset('targets', data=np.array(targets, dtype=h5py.special_dtype(vlen=str)))
                    f1.create_dataset('sents', data=np.array(sents, dtype=h5py.special_dtype(vlen=str)))
                    f1.close()

            # calculate accuracy
            if self.calc_acc:
                audio_logits =  (audio_batch @ text_batch.t()) * self.fixed_scale
                text_logits = audio_logits.t()
                audio_probs = audio_logits.softmax(dim=-1).cpu().numpy()
                text_probs = text_logits.softmax(dim=-1).cpu().numpy()
                ground_truth = torch.arange(self.bs,  dtype=torch.long).cpu()
                audio_acc = torch.eq(torch.tensor(audio_probs.argmax(axis=0)), ground_truth).sum() / ground_truth.shape[0]
                text_acc = torch.eq(torch.tensor(text_probs.argmax(axis=0)), ground_truth).sum() / ground_truth.shape[0]
                accs.append((audio_acc.item() + text_acc.item()) / 2)

            cur_idx = step * self.bs
            next_idx = cur_idx + len(targets)
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

        if self.calc_acc:
            print("Final accuracy: ", np.mean(accs))
            self.acc = np.mean(accs)

    def encode_queries(self, topics_df, query_field='query'):
        
        if query_field == 'query':
            embed_path = conf.yamnet_query_embed_path
            # Read queries
            self.queries = topics_df[query_field].tolist()
            self.query_text_encoding = self.text_to_embed(self.queries)
        
        elif query_field == 'description':
            embed_path = conf.yamnet_descr_embed_path
            self.query_descr = topics_df[query_field].tolist()
            self.descr_text_encoding = self.text_to_embed(self.query_descr)
        self.query_nums  = topics_df['num'].tolist()

        # Read yamnet embeddings for queries
        query_yamnets = []
        query_lengths = []
        for idx, row in topics_df.iterrows():
            query_num = row.num
            query_embed_path  = os.path.join(embed_path, str(query_num) + ".h5")
            inbetween_yamnet_embeds = pd.read_hdf(query_embed_path)

            query_lengths.append(len(inbetween_yamnet_embeds))
            tmp = torch.Tensor(inbetween_yamnet_embeds.values)
            query_yamnets.append(tmp)

        # padded_audio_embeds = pad_sequence(audio_embeds, batch_first=True).to(self.device)
            
        if query_field == 'query':
            # Create query embeddings
            self.query_audio_encoding = self.audio_to_embed(query_yamnets, query_lengths).cpu()
            print("Queries encoded ", self.query_audio_encoding.shape, self.query_text_encoding.shape)
        elif query_field == 'description':
            self.descr_audio_encoding = self.audio_to_embed(query_yamnets, query_lengths).cpu()
            print("Descriptions encoded ", self.query_audio_encoding.shape, self.query_text_encoding.shape)

    def add_query_labels(self, topics_df, val_df):
        query_nums  = topics_df['num'].tolist()
        relevant_segs = defaultdict(list)
        relevant_eps = {}
        for query_idx, query_num in enumerate(query_nums):
            relevant = val_df[(val_df.num == query_num) & (val_df.bin_score == 1)]
            relevant_segs[query_idx] = list(dict.fromkeys(relevant.episode_uri_time))
            relevant_eps[query_idx] = list(dict.fromkeys(relevant.episode_uri))
        self.relevant_segs = relevant_segs
        self.relevant_eps = relevant_eps
    
def evaluate_topk(evaluator, query_encodings, pod_encodings):
    full_results = defaultdict(list)
    k = 250
    pred =[[1]* 50 + [0] * 50]
    
    targets = evaluator.all_targs
    relevant_segs = evaluator.relevant_segs
    relevant_eps = evaluator.relevant_eps
    
    for query_tup in query_encodings:
        for tup in pod_encodings:
            name = query_tup[1] + "2" + tup[1]
            print("------- Results for: ", name)
            query_encoding = query_tup[0]
            pod_encoding = tup[0]

            print("[del] query: ", query_encoding.is_cuda)
            # print("[del] pod_encoding: ", pod_encoding.is_cuda)

            full_results[name] = defaultdict(list)
            similarity = (100.0 * query_encoding @ pod_encoding.T).softmax(dim=-1)
            
            for query_idx in range(len(query_encoding)):
                values, indices = similarity[query_idx].topk(k)
                
                num_rel = len(relevant_eps[query_idx])
                num_rel_segs = len(relevant_segs[query_idx])
                
                # If we have no relevance labels, do not calculate metrics
                if num_rel == 0:
                    continue

                predicted_segs = list(dict.fromkeys([targets[i] for i in indices]))
                predicted_epis = list(dict.fromkeys([i.split("_")[0] for i in predicted_segs]))
                
                if len(predicted_epis) < 100:
                    print("TODO: increase K above")
                    print(len(predicted_epis))
                
                scores_epis = [1 if e in relevant_eps[query_idx] else 0 for e in predicted_epis]
                scores_segs = [1 if e in relevant_segs[query_idx] else 0 for e in predicted_segs]

                ndcg_ep = ndcg_score(pred, [scores_epis[:100]])
                p10_ep = prec_at_k(scores_epis, k=10)
                p10_seg = prec_at_k(scores_segs, k=10)
                p100_ep = prec_at_k(scores_epis, k=100)
                r100_ep = rec_at_k(scores_epis, num_rel, k=100)
                r100_seg = rec_at_k(scores_segs, num_rel_segs, k=100)
                print("Metrics: ", p10_ep, p10_seg, r100_ep)
                
                if p10_ep > 0.5:
                    print("Good results for query: ", evaluator.queries[query_idx])
                    for i in range(10):
                        print(evaluator.all_sents[indices[i]])
                    print(scores_epis)
                
                full_results[name]['p10_ep'].append(p10_ep)
                full_results[name]['p10_seg'].append(p10_seg)
                full_results[name]['p100_ep'].append(p100_ep)
                full_results[name]['r100_ep'].append(r100_ep)
                full_results[name]['r100_seg'].append(r100_seg) 
                full_results[name]['ndcg_ep'].append(ndcg_ep) 
                
    return full_results

def save_eval_results(full_results, evaluator):
    mean_results = {}
    for name, metric in full_results.items():
        mean_results[name] = {}
        for metric_name, vals in metric.items():
            mean_results[name][metric_name] = np.mean(vals)

    mean_results['test_acc'] = evaluator.acc
    out = os.path.join(evaluator.model_path, "topictask_results.json")
    with open(out, "w") as f:
        json.dump(mean_results, f, indent=4)
    out = os.path.join(evaluator.model_path, "topictask_results.csv")
    df = pd.DataFrame(mean_results)
    df.to_csv(out )
    print(df.T.to_string())


def main(args):
    print("[main] Evaluate")
    # Reading the metadata.
    metadata_testset, topics_df, topics_df_targets = read_metadata_subset(conf, traintest='test')
    print(len(metadata_testset))

    # Remove duplicate rows
    metadata_testset = metadata_testset.drop_duplicates(subset=['episode_filename_prefix']).sort_values('episode_filename_prefix')
    print("[main] Topic test set loaded: ", len(metadata_testset))

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
    print("[Load model] config loaded: ", CFG)

    # Create dataloader of test set
    data_loader = MMloader(CFG)

    # Load the model
    full_model = mmModule(CFG)
    full_model.load_state_dict(torch.load(model_weights_path,  map_location=CFG.device))              
    full_model = full_model.to(CFG.device)     
    full_model.eval()

    # Calculate embeddings for test set
    evaluator = Evaluator(CFG, model_path, full_model, data_loader, 
            args.save_intermediate, args.calc_acc)
    max_samples = evaluator.get_max_data()

    # max_samples = 128 * 400
    # print("deleteeeee")

    evaluator.encode_testset_new(max_samples)
    evaluator.encode_queries(topics_df, query_field='query')
    evaluator.encode_queries(topics_df, query_field='description')
    evaluator.add_query_labels(topics_df, val_df)

    # Now perform evaluation. 
    query_encodings = [(evaluator.query_text_encoding, 'querytext'),
                    (evaluator.query_audio_encoding, 'queryaudio'),
                    (evaluator.descr_text_encoding, 'descrtext'),
                    (evaluator.descr_audio_encoding, 'descraudio')]
    pod_encodings = [
                    (evaluator.text_encoding, 'text'), 
                    (evaluator.audio_encoding, 'audio')]

    full_results = evaluate_topk(evaluator, query_encodings, pod_encodings)


    # Save results      # TODO: seperate function
    save_eval_results(full_results, evaluator)
    

if __name__ == "__main__":
    print("[evaluate] reading data from: ")
    print(conf.dataset_path)

    # Parse flags in command line arguments
    parser = ArgumentParser()

    parser.add_argument('--model_path', type=str, default="logs/run2-clip_loss_None_gru_768_False_2022-06-12_12-41-49",
                        help='Folder where model weights are saved.')

    
    parser.add_argument('--save_intermediate', action='store_true', default=False,
                    help='Whether to save intermediate embeddings.')
    # parser.add_argument('--tmp_files_path', type=str, default="logs/windows_gru2_15m",
    #                     help='Folder where model weights are saved.')
    parser.add_argument('--calc_acc', action='store_true', default=False,
                    help='Whether to output accuracy.')
    args, unparsed = parser.parse_known_args()

    main(args)
