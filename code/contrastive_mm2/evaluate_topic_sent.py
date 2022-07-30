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
import gc

from argparse import ArgumentParser
from dacite import from_dict
from omegaconf import OmegaConf
from torch import topk
from sklearn.metrics import ndcg_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

import torch 
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from nltk import tokenize

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

def rec_at_k_norm(scores, num_rel, k):
    # (of recommended items that are relevant @k)/(total # of relevant items)
    scores_k = scores[:k]
    rec = np.sum(scores_k) / num_rel 
    return rec

def rec_at_k(scores, k):
    # (of recommended items that are relevant @k)/(total # of relevant items)
    scores_k = scores[:k]
    score = np.sum(scores_k) 
    if score >= 1:
        return 1
    else:
        return 0

def randomize_model(model):
    for module_ in model.named_modules(): 
        if isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
            module_[1].weight.data.normal_(mean=0.0, std=1.0)
        elif isinstance(module_[1], torch.nn.LayerNorm):
            module_[1].bias.data.zero_()
            module_[1].weight.data.fill_(1.0)
        if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
            module_[1].bias.data.zero_()
    return model



def randomize_model(model):
    for module_ in model.named_modules(): 
        if isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
            module_[1].weight.data.normal_(mean=0.0, std=1.0)
        elif isinstance(module_[1], torch.nn.LayerNorm):
            module_[1].bias.data.zero_()
            module_[1].weight.data.fill_(1.0)
        if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
            module_[1].bias.data.zero_()
    return model


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
        #self.bs = args.batch_size
        self.bs = 128
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
        max_samples =  len(self.test_loader) * self.bs
        return max_samples

    def audio_to_embed(self, yamnets, query_lengths):
        if self.audio_proj_head in ['gru', 'gru_v2', 'rnn', 'lstm', 'mlp']:
            padded_yamnets = pad_sequence(yamnets, batch_first=True).to(self.device)

            with torch.no_grad():
                reps_audio = self.model.audio_model((padded_yamnets, query_lengths))
                # embed = reps_audio / reps_audio.norm(dim=1, keepdim=True)
                embed = torch.nn.functional.normalize(reps_audio) 
        else:
            with torch.no_grad():

                yamnets_mean = [torch.tensor(y.mean(dim=0).clone().detach()) for y in yamnets]

                audio_features = torch.stack(yamnets_mean).to(self.device)
                reps_audio = self.model.audio_model((audio_features, query_lengths))
                # embed = reps_audio / reps_audio.norm(dim=1, keepdim=True)
                embed = torch.nn.functional.normalize(reps_audio) 
        return embed.cpu()

    def text_to_embed(self, text):
        tokenized_text = self.tokenizer(
            text, padding=True, truncation=True, max_length=32, return_tensors='pt', return_token_type_ids=True,
        )
        
        with torch.no_grad():
            tokenized_text = tokenized_text.to(self.device)
            reps_sentences = self.model.text_model(tokenized_text)['sentence_embedding']
            # embed = reps_sentences / reps_sentences.norm(dim=1, keepdim=True)
            embed = torch.nn.functional.normalize(reps_sentences) 

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

                reps_text = self.model.text_model(tok_sentences)['sentence_embedding']
                reps_audio = self.model.audio_model((audio_features, seq_len))

                audio_batch = torch.nn.functional.normalize(reps_audio) 
                text_batch = torch.nn.functional.normalize(reps_text) 


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
            targets = [t.split(".")[0] for t in targets]
            all_targs[cur_idx:next_idx] = targets

            if next_idx >= max_samples: 
                print("Max samples reached")
                break
          
        self.all_sents = all_sents
        self.all_targs = all_targs
        self.text_encoding = torch.tensor(text_encoding #hiero
        self.audio_encoding = audio_encoding

        if self.calc_acc:
            print("Final accuracy: ", np.mean(accs))
            self.mean_acc = np.mean(accs)
            self.std_acc = np.std(accs)
            self.var_acc = np.var(accs)

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
            
    def encode_sent_descr(self, topics_df, field, sent_dict, audio_dict):
        print("encode sent descr!")
        
        targets = []
        text_encodings = []
        texts = []
        audio_encodings = []
        
        for epi, full_text in sent_dict.items():
            audio_embeds = audio_dict[epi]
            #print("embeds: ", audio_embeds)

            targets.append(epi)
                
            tokenized_text = tokenize.sent_tokenize(full_text)  # for now only first sentence
            text_encoding = self.text_to_embed(tokenized_text)[0]
            text_encodings.append(text_encoding)
            texts.append(tokenized_text)

            query_length = len(audio_embeds[0])
            audio_encoding = self.audio_to_embed([audio_embeds[0]], [query_length]).cpu()
            audio_encodings.append(audio_encoding)
            #print("Texts: ", texts)

        if field == 'query':
            # Create query embeddings
            self.sent_topic_query_targets = targets
            self.sent_topic_query_text_encoding = torch.stack(text_encodings)
            self.sent_topic_query_texts = texts
            self.sent_topic_query_audio_encoding = torch.vstack(audio_encodings)


        elif field == 'description':
            self.sent_topic_descr_targets = targets
            self.sent_topic_descr_text_encoding = torch.stack(text_encodings)
            self.sent_topic_descr_texts = texts
            self.sent_topic_descr_audio_encoding = torch.vstack(audio_encodings)
        print("Done eoncoding dexr, field: ", field)
        
    def add_query_labels(self, topics_df, val_df):
        query_nums  = topics_df['num'].tolist()
        relevant_segs = defaultdict(list)
        relevant_eps = {}
        
        episode_targets = {}
        for query_idx, query_num in enumerate(query_nums):
            relevant = val_df[(val_df.num == query_num) & (val_df.bin_score == 1)]
            relevant_segs[query_idx] = list(dict.fromkeys(relevant.episode_uri_time))
            relevant_eps[query_idx] = list(dict.fromkeys(relevant.episode_uri))
            
            episode_targets[query_num] = list(dict.fromkeys(relevant.episode_uri))
        self.relevant_segs = relevant_segs
        self.relevant_eps = relevant_eps
        
        self.episode_targets = episode_targets

def get_sent_audio(sent_summary_embed_dir):
    # Read summary embeddings into a dict
    sent_summary_audio_dict = defaultdict(list)

    pathlist = Path(sent_summary_embed_dir).glob('**/*.h5')
    for path in pathlist:
        # because path is object not string
        filename = str(path)

        with h5py.File(filename, "r") as f:
            filename = os.path.split(filename)[-1]
            target_name = filename.split('.')[0]
            emb = torch.Tensor(f['embedding']['block0_values'])
            sent_summary_audio_dict[target_name].append(emb)

    return sent_summary_audio_dict


# def save_eval_results(full_results, evaluator):
#     mean_results = {}
#     for name, metric in full_results.items():
#         mean_results[name] = {}
#         for metric_name, vals in metric.items():
#             mean_results[name][metric_name] = np.mean(vals)

#     mean_results['test_mean_acc'] = evaluator.mean_acc
#     mean_results['test_std_acc'] = evaluator.std_acc
#     mean_results['test_var_acc'] = evaluator.var_acc

#     # out_path = os.path.join(evaluator.model_path, "topic_evaluation")
#     name = os.path.split(args.model_weights_path)[-1].split(".")[0]
#     # Path(out_path).mkdir(parents=True, exist_ok=True)

#     out = os.path.join(evaluator.model_path, "topic_results_{}.json".format(name))
#     with open(out, "w") as f:
#         json.dump(mean_results, f, indent=4)
#     out = os.path.join(evaluator.model_path, "topic_results_{}.csv".format(name))
#     df = pd.DataFrame(mean_results)
#     df.T.to_csv(out, sep=';')
#     print(df.T.to_string())

def topic_evaluation(evaluator):
    ### Step 1B: Results on sent-summary level, merging all sentences of summary

    topic_encodings = [(evaluator.sent_topic_descr_text_encoding, 'sent_descr_text'),
                    (evaluator.sent_topic_descr_audio_encoding, 'sent_descr_audio'),
                    #(evaluator.sent_topic_query_text_encoding, 'sent_query_text'),
                    #(evaluator.sent_topic_query_audio_encoding, 'sent_query_audio'),
                        ]
    targets = evaluator.sent_topic_descr_targets
    texts = evaluator.sent_topic_descr_texts

    epi_encodings = [(evaluator.text_encoding, 'text'),
                    (evaluator.audio_encoding, 'audio')]

    k = evaluator.text_encoding.shape[0]
    results = defaultdict(list)
    for topic_tup in topic_encodings:
        
        for tup in epi_encodings:
            # gc.collect()
            # print("[del] collected memory")
            name = topic_tup[1] + "2" + tup[1]
            print("------- Results for: ", name)
            topic_encoding = topic_tup[0]
            epi_encoding = tup[0]
            
            print("[del] before previous error: ", type(topic_encoding), type(epi_encoding))
            print("[del2]:  ", topic_encoding.dtype, epi_encoding.dtype)
            exit(1)
            
            similarity = (100.0 * topic_encoding @ epi_encoding.T).softmax(dim=-1)
            # print(type(similarity))
            # exit(1)
            rank = []        
            mrr = []
            
            confidence = []
            total_indices = []
            idxs= []
            for idx in range(len(targets)):
                print("k={}, Idx: {}/{}".format(k, idx, len(targets)))
                values, indices = similarity[idx].topk(k)
                target = targets[idx].split("_")[0]
                
                confidence.extend(values)
                total_indices.extend(indices)
                idxs.append(idx)
                    
                if idx != (len(targets) - 1):
                    next_target = targets[idx+1].split("_")[0]
                    if next_target == target:
                        continue
                    
                confidence = np.array(confidence)
                total_indices = np.array(total_indices)
                sorted_inds = confidence.argsort()

                sorted_inds = confidence.argsort()[::-1]
                total_indices = total_indices[sorted_inds]

                predicted_segs = evaluator.all_targs[total_indices.tolist()].tolist()
                predicted_epis = list(dict.fromkeys([i.split("_")[0] for i in predicted_segs]))

                # OPTION 1: on episode level
                ranks = []
                for possible_target in evaluator.episode_targets[int(target)]:
                    if possible_target in predicted_epis:
                        estimated_position = predicted_epis.index(possible_target)
                        ranks.append( estimated_position)
                        print("I predicted an episode on rank: ", estimated_position)
                if len(ranks) > 0:
                    best_estimation = np.min(ranks)
                    rank.append(best_estimation)
                
                    mrr.append(1 / (best_estimation + 1))
                    
                    print("++ Estimated position: ", best_estimation, confidence[sorted_inds[0]])
                    if best_estimation < 1:
                        for i in idxs:
                            print("         summary: ", texts[i])
                        print("          pred: ", evaluator.all_sents[total_indices[0].tolist()])
                
                else:
                    print("-- Target not in list?")
                confidence = []
                total_indices = []
                idxs= []

            print("[del] done")
            print("Mean position: {} (std: {}) \t mrr: {} ".format(np.mean(rank), np.std(rank), np.mean(mrr)))
            results['names'].append(name)
            results['ranks'].append(np.mean(rank))
            results['ranks_var'].append(np.var(rank))
            results['ranks_std'].append(np.std(rank))
            results['mrrs'].append(np.mean(mrr))
            results['mrrs_var'].append(np.std(mrr))
            results['mrrs_std'].append(np.var(mrr))

            del similarity
    return results

def main(args):
    print("[main] Evaluate")

    sent_query_output_path = os.path.join(conf.dataset_path, 'TREC_topic', 'topic_dict_query.json')
    sent_descr_output_path = os.path.join(conf.dataset_path, 'TREC_topic', 'topic_dict_descr.json')

    ## Read dictionary with summaries
    with open(sent_query_output_path, 'r') as f:
        sent_query_dict = json.load(f)
    with open(sent_descr_output_path, 'r') as f:
        sent_descr_dict = json.load(f)
    print(sent_descr_dict.keys())
    sent_descr_audio_dict = get_sent_audio(conf.sent_topic_descr_embed_dir)

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
    model_path = Path(args.model_weights_path).parents[1]
    # model_path = Path('logs/30m-gru_v2_2022-07-06_07-25-51/output/full_model_weights.pt').parents[1]
    model_weights_path = args.model_weights_path
    # model_weights_path = 'logs/30m-gru_v2_2022-07-06_07-25-51/output/full_model_weights.pt'
    model_config_path = os.path.join(model_path, 'config.json')

    # Opening JSON file
    f = open(model_config_path)
    model_config = json.load(f)
    print("using config: , ", model_config)
    CFG = from_dict(data_class=Cfg, data=model_config)
    CFG.device = "cuda" if torch.cuda.is_available() else "cpu"
    CFG.sp_path = conf.sp_path
    print("[Load model] config loaded: ", CFG)

    # Create dataloader of test set
    data_loader = MMloader(CFG)

    # Load the model
    full_model = mmModule(CFG)
    full_model.load_state_dict(torch.load(model_weights_path,  map_location=CFG.device))   
    # To create random results:
    # full_model = randomize_model(full_model) # REMOVE THIS!!

    full_model = full_model.to(CFG.device)     
    full_model.eval()

    # Calculate embeddings for test set
    evaluator = Evaluator(CFG, model_path, full_model, data_loader, 
            save_intermediate=False, calc_acc = False)
    max_samples = evaluator.get_max_data()

    max_samples = 128 * 5
    print("deleteeeee")

    evaluator.encode_testset_new(max_samples) 
    # evaluator.encode_queries(topics_df, query_field='query')
    # evaluator.encode_queries(topics_df, query_field='description')
    evaluator.encode_sent_descr(topics_df, field='description', sent_dict=sent_descr_dict, audio_dict=sent_descr_audio_dict)
    # evaluator.encode_sent_descr(topics_df, field='query', sent_dict=sent_query_dict, audio_dict=sent_query_audio_dict)

    evaluator.add_query_labels(topics_df, val_df)

    results = topic_evaluation(evaluator)

    # save the results
    json_out = str(Path(args.model_weights_path).parents[1])
    with open(os.path.join(json_out, 'topic_sent_results.json'), 'w') as fp:
        json.dump(results, fp, indent=4)

    out = os.path.join(evaluator.model_path, "topic_sent_results.csv")
    df = pd.DataFrame(results)
    df.T.to_csv(out, sep=';')
    print(df.T.to_string())


    

if __name__ == "__main__":
    print("[evaluate] reading data from: ")
    print(conf.dataset_path)

    # Parse flags in command line arguments
    parser = ArgumentParser()

    parser.add_argument('--model_weights_path', type=str, default="logs/30m-gru_v2_2022-07-06_07-25-51/output/full_model_weights.pt",
                        help='Folder where model weights are saved.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size to use.')
    parser.add_argument('--save_intermediate', action='store_true', default=False,
                    help='Whether to save intermediate embeddings.')
    parser.add_argument('--calc_acc', action='store_true', default=False,
                    help='Whether to output accuracy.')
    args, unparsed = parser.parse_known_args()

    main(args)
