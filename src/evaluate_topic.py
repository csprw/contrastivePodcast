"""
Self-supervised Contrastive Learning from Podcast Audio and Transcripts.
Topic evaluation task.
Author: Casper Wortmann
Usage: python evaluate_topic.py
"""
import h5py
import json
import os
from pathlib import Path
import math
import pandas as pd
import numpy as np
import glob
from collections import defaultdict
from argparse import ArgumentParser
from dacite import from_dict
from omegaconf import OmegaConf

from torch import topk
import torch 
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from nltk import tokenize
from utils import read_metadata_subset, randomize_model
from train import mmModule, Cfg
from train import MMloader

class Evaluator(object):
    def __init__(self, CFG, model_path, full_model, data_loader, calc_acc):
        """
        Evaluator object to perform a topical ranking task. 
        Inputs:
            CFG: configuration of the current model. 
            model_path: path to the run we want to evaluate. 
            full_model: the model with the loaded weights. 
            data_loader: a summary dataoader. 
            calc_acc: bool. If set to true, also calculate prediction accuracy.
        """
        self.model_path = model_path
        self.model = full_model
        self.device = CFG.device
        self.scale = CFG.scale
        self.bs = CFG.batch_size
        self.embed_dim = CFG.mutual_embedding_dim
        self.tokenizer =  data_loader.test_dataset.tokenizer
        self.test_loader = data_loader.test_loader
        self.test_dataset = data_loader.test_dataset
        self.CFG = CFG
        self.audio_proj_head = CFG.audio_proj_head
        self.fixed_scale = CFG.scale
        
        self.test_dir = os.path.join(CFG.sp_path, 'test')
        self.calc_acc = calc_acc
        self.acc = 0.0

    def get_max_data(self):
        """
        Returns max number of sentences to encode.
        """
        max_samples = self.bs  * math.floor(len(self.test_dataset)/self.bs)
        return max_samples

    def audio_to_embed(self, yamnets, query_lengths):
        """
        Creates representations of all (precomputed) yamnet embeddings of the
        episodes in the topic-test set of the SP dataset.
        Inputs:
            yamnets: list of precomputed yamnet embeddings.
            query_lengths: lengths of the yamnet embeddings before padding. 
        Outputs:
            embed: an audio embedding. 
        """
        if self.audio_proj_head in ['gru', 'rnn', 'lstm', 'mlp']:
            # If the audio projection head contains an RNN-based layer we can
            # push all yamnets through it.
            padded_yamnets = pad_sequence(yamnets, batch_first=True).to(self.device)

            with torch.no_grad():
                reps_audio = self.model.audio_model((padded_yamnets, query_lengths))
                embed = torch.nn.functional.normalize(reps_audio) 
        else:
            with torch.no_grad():
                # Take the mean of the yamnet embeddings and push through SPH. 
                yamnets_mean = [torch.tensor(y.mean(dim=0).clone().detach()) for y in yamnets]
                audio_features = torch.stack(yamnets_mean).to(self.device)
                reps_audio = self.model.audio_model((audio_features, query_lengths))
                embed = torch.nn.functional.normalize(reps_audio) 
        return embed.cpu()

    def text_to_embed(self, text):
        """
        Creates embeddings of a text-string. 
        Inputs:
            text: a list of sentences.
        Outputs:
            embed: text embedding. 
        """
        tokenized_text = self.tokenizer(
            text, padding=True, truncation=True, max_length=32, 
            return_tensors='pt', return_token_type_ids=True,
        )
        
        with torch.no_grad():
            tokenized_text = tokenized_text.to(self.device)
            reps_sentences = self.model.text_model(tokenized_text)['sentence_embedding']
            embed = torch.nn.functional.normalize(reps_sentences) 

        return embed.cpu()

    @torch.no_grad()  
    def encode_testset(self, max_samples):
        """
        Creates embeddings of all queries, descriptions, and episodes in the topic-test set.
        Inputs:
            max_samples: the maximum number of samples to encode 
            (mostly used for debugging and the notebooks). 
        """
        max_steps = int(max_samples/self.bs)
        accs = []
        audio_accs = []
        text_accs = []
        text_encoding = np.zeros((max_samples, self.embed_dim)) 
        audio_encoding = np.zeros((max_samples, self.embed_dim)) 
        
        all_sents = np.zeros(max_samples, dtype=object)
        all_targs = np.zeros(max_samples, dtype=object)
        
        for step, (tok_sentences, audio_features, seq_len, targets, sents) in enumerate(iter(self.test_loader)):
            print("Calculating: ", step, max_steps)
            tok_sentences = tok_sentences.to(self.device)
            audio_features = audio_features.to(self.device)  

            reps_text = self.model.text_model(tok_sentences)['sentence_embedding']
            reps_audio = self.model.audio_model((audio_features, seq_len))

            audio_batch = torch.nn.functional.normalize(reps_audio) 
            text_batch = torch.nn.functional.normalize(reps_text) 

            # Calculate accuracy metrics.
            if self.calc_acc:
                audio_logits =  (audio_batch @ text_batch.t()) * self.fixed_scale
                text_logits = audio_logits.t()
                audio_probs = audio_logits.softmax(dim=-1).cpu().numpy()
                text_probs = text_logits.softmax(dim=-1).cpu().numpy()
                ground_truth = torch.arange(self.bs,  dtype=torch.long).cpu()
                audio_acc = torch.eq(torch.tensor(audio_probs.argmax(axis=0)), ground_truth).sum() / ground_truth.shape[0]
                text_acc = torch.eq(torch.tensor(text_probs.argmax(axis=0)), ground_truth).sum() / ground_truth.shape[0]
                accs.append((audio_acc.item() + text_acc.item()) / 2)
                audio_accs.append(audio_acc.item())
                text_accs.append(text_acc.item())

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
        self.text_encoding = torch.tensor(text_encoding, dtype=torch.float32)
        self.audio_encoding = torch.tensor(audio_encoding, dtype=torch.float32)

        if self.calc_acc:
            print("Final accuracy: ", np.mean(accs))
            self.mean_acc = np.mean(accs)
            self.std_acc = np.std(accs)
            self.var_acc = np.var(accs)

            self.audio_acc = np.mean(audio_accs)
            self.std_audio_acc = np.std(audio_accs)
            self.var_audio_acc = np.var(audio_accs)

            self.text_acc = np.mean(text_accs)
            self.std_text_acc = np.std(text_accs)
            self.var_text_acc = np.var(text_accs)

    def encode_queries(self, topics_df, query_field='query'):
        """
        Create representations of the queries in the topic-test set. 
        Inputs: 
            topics_df: Dataframe containing queries and descriptions.
            query_field: Wheter to encode the queries or the query descriptions.
        """
        assert query_field in [ 'query', 'description']
        if query_field == 'query':
            embed_path = self.CFG.yamnet_query_embed_path
            self.queries = topics_df[query_field].tolist()
            self.query_text_encoding = self.text_to_embed(self.queries)
        
        elif query_field == 'description':
            embed_path = self.CFG.yamnet_descr_embed_path
            self.query_descr = topics_df[query_field].tolist()
            self.descr_text_encoding = self.text_to_embed(self.query_descr)
        self.query_nums  = topics_df['num'].tolist()

        # Read yamnet embeddings for queries
        query_yamnets = []
        query_lengths = []
        for _, row in topics_df.iterrows():
            query_num = row.num
            query_embed_path  = os.path.join(embed_path, str(query_num) + ".h5")
            inbetween_yamnet_embeds = pd.read_hdf(query_embed_path)

            query_lengths.append(len(inbetween_yamnet_embeds))
            tmp = torch.Tensor(inbetween_yamnet_embeds.values)
            query_yamnets.append(tmp)

        # Create represntations from the yamnet embeddings. 
        if query_field == 'query':
            self.query_audio_encoding = self.audio_to_embed(query_yamnets, query_lengths).cpu()
        elif query_field == 'description':
            self.descr_audio_encoding = self.audio_to_embed(query_yamnets, query_lengths).cpu()
            
    def encode_sent_descr(self, sent_dict, audio_dict):
        """
        Create representations for all sentences in the descriptions of queries. 
        Inputs:
            sent_dict: a dictionary with the text-sentences of the descriptions. 
            audio_dict: a dictionary with the audio-sentences of the descriptions. 
        """
        targets = []
        text_encodings = []
        texts = []
        audio_encodings = []
        
        for epi, full_text in sent_dict.items():
            audio_embeds = audio_dict[epi]
            targets.append(epi)
                
            tokenized_text = tokenize.sent_tokenize(full_text)  # for now only first sentence
            text_encoding = self.text_to_embed(tokenized_text)[0]
            text_encodings.append(text_encoding)
            texts.append(tokenized_text)

            query_length = len(audio_embeds[0])
            audio_encoding = self.audio_to_embed([audio_embeds[0]], [query_length]).cpu()
            audio_encodings.append(audio_encoding)

        self.sent_topic_descr_targets = targets
        self.sent_topic_descr_text_encoding = torch.stack(text_encodings).type(torch.float32)
        self.sent_topic_descr_texts = texts
        self.sent_topic_descr_audio_encoding = torch.vstack(audio_encodings).type(torch.float32)
        
    def add_query_labels(self, topics_df, val_df):
        """
        Annotates all segments and episodes in the test-set as to whether they
        are relevant according to the relevance assesments.
        Inputs:
            topics_df: a df containing the topic descriptions.
            val_df: a df containing the topic assessments. 
        """
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

    def encode_queries_natural(self, query_field):
        """
        Encode the audio for the query-description similarity task. 
        Inputs: 
            query_field: whether to encode queries or descriptions. 
        """
        assert query_field in [ 'query', 'descr']
        if query_field == 'descr':
            sent_topic_descr_embed_dir = self.CFG.sent_topic_descr_embed_dir
            embeddings = {}
            field_files = sorted(glob.glob(os.path.join(sent_topic_descr_embed_dir,'*.h5')))
            
        if query_field == 'query':
            sent_topic_query_embed_dir = self.CFG.sent_topic_query_embed_dir
            embeddings = {}
            field_files = sorted(glob.glob(os.path.join(sent_topic_query_embed_dir,'*.h5')))

        # Iterate over embeddings and compute audio encodings. 
        for embed_path in field_files:
            inbetween_yamnet_embeds = pd.read_hdf(embed_path)
            yamnet_len = len(inbetween_yamnet_embeds)
            yamnet_embed = torch.Tensor(inbetween_yamnet_embeds.values)
            audio_encoding = self.audio_to_embed([yamnet_embed], [yamnet_len]).cpu()
            name = os.path.split(embed_path)[-1].split(".")[0]
            embeddings[name] = audio_encoding

        if query_field == 'descr':
            self.embed_descr_natural = embeddings
        elif query_field == 'query':
            self.embed_query_natural = embeddings

    def encode_queries_descriptions(self, topics_df):
        """ Encode the text for the query-description similarity task. """
        ds =  topics_df['description'].tolist()
        ds_persent = [tokenize.sent_tokenize(p) for p in ds]
        descr_text_sent_encoding = self.text_to_embed([d[0] for d in ds_persent])
        self.descr_text_sent_encoding = descr_text_sent_encoding
        

def get_sent_audio(embed_dir):
    """
    Calculates audio embeddings. 
    Inputs:
        embed_dir: path to the precomputed yamnet embeddings of descriptions. 
    Outputs:
        sent_summary_audio_dict: the audio embeddings for each description-sentence. 
    """
    sent_summary_audio_dict = defaultdict(list)

    pathlist = Path(embed_dir).glob('**/*.h5')
    for path in pathlist:
        filename = str(path)

        with h5py.File(filename, "r") as f:
            filename = os.path.split(filename)[-1]
            target_name = filename.split('.')[0]
            emb = torch.Tensor(np.array(f['embedding']['block0_values']))
            sent_summary_audio_dict[target_name].append(emb)

    return sent_summary_audio_dict

def topic_evaluation(evaluator):
    """
    Creates similarity matrices between the topic encodings and 
    episode encodings. Based on this similarity matrix a ranking is created,
    and validated.
    Inputs:
        evaluator: an instance of Evaluator class.
    """
    topic_encodings = [(evaluator.sent_topic_descr_text_encoding, 'sent_descr_text'),
                    (evaluator.sent_topic_descr_audio_encoding, 'sent_descr_audio')]
    targets = evaluator.sent_topic_descr_targets
    texts = evaluator.sent_topic_descr_texts

    epi_encodings = [(evaluator.text_encoding, 'text'),
                    (evaluator.audio_encoding, 'audio')]

    k = evaluator.text_encoding.shape[0]
    #k = min(5000, evaluator.text_encoding.shape[0]) # In case of memory problems. 
    results = defaultdict(list)

    for topic_tup in topic_encodings:
        for tup in epi_encodings:
            name = topic_tup[1] + "-" + tup[1]
            print("------- Results for: ", name)
            topic_encoding = topic_tup[0]
            epi_encoding = tup[0]

            # To prevent memory problems, we create a similarity matrix in batches. 
            bound = int(epi_encoding.shape[0] / 8)
            start_bound = 0
            cur_bound = bound
            sims = []
            for _ in range(7):
                sim = (100.0 * topic_encoding @ epi_encoding[start_bound:cur_bound].T)
                start_bound = cur_bound
                cur_bound += bound
                sims.append(sim)
            sim = (100.0 * topic_encoding @ epi_encoding[start_bound:cur_bound].T)
            sims.append(sim)
            similarity = (100.0 * topic_encoding @ epi_encoding.T)
            similarity = torch.hstack(sims)
            del sims

            # Transform similarity matrix to probabilites. 
            similarity = similarity.softmax(dim=-1)
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
                    
                # Wait untill we have processed all sentences of a target. 
                if idx != (len(targets) - 1):
                    next_target = targets[idx+1].split("_")[0]
                    if next_target == target:
                        continue
                    
                # Sort the rankings. 
                confidence = np.array(confidence)
                total_indices = np.array(total_indices)
                sorted_inds = confidence.argsort()[::-1]
                total_indices = total_indices[sorted_inds]

                # Get the positions of the predicted episodes. 
                predicted_segs = evaluator.all_targs[total_indices.tolist()].tolist()
                predicted_epis = list(dict.fromkeys([i.split("_")[0] for i in predicted_segs]))

                # Calculate ranks. 
                ranks = []
                for possible_target in evaluator.episode_targets[int(target)]:
                    if possible_target in predicted_epis:
                        estimated_position = predicted_epis.index(possible_target)
                        ranks.append( estimated_position)
                confidence = []
                total_indices = []
                idxs= []

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

def save_eval_results(evaluator, results):
    """ Saves the results of the evaluation to local disk. """
    results['mean_acc'] = evaluator.mean_acc
    results['std_acc'] = evaluator.std_acc
    results['var_acc'] = evaluator.var_acc

    results['audio_acc'] = evaluator.audio_acc
    results['std_audio_acc'] = evaluator.std_audio_acc
    results['var_acc'] = evaluator.var_audio_acc

    results['text_acc'] = evaluator.text_acc
    results['std_text_acc'] = evaluator.std_text_acc
    results['var_text_acc'] = evaluator.var_text_acc

    json_out = str(Path(args.model_weights_path).parents[1])
    with open(os.path.join(json_out, 'topic_sent_results.json'), 'w') as fp:
        json.dump(results, fp, indent=4)

    out = os.path.join(evaluator.model_path, "topic_sent_results.csv")
    df = pd.DataFrame(results)
    df.T.to_csv(out, sep=';')
    print(df.T.to_string())

def main(args):
    print("[main] Evaluate")
    conf = OmegaConf.load("./config.yaml")
    sent_query_output_path = os.path.join(conf.dataset_path, 'TREC_topic', 'topic_dict_query.json')
    sent_descr_output_path = os.path.join(conf.dataset_path, 'TREC_topic', 'topic_dict_descr.json')

    ## Read dictionary with queries
    with open(sent_query_output_path, 'r') as f:
        sent_query_dict = json.load(f)
    with open(sent_descr_output_path, 'r') as f:
        sent_descr_dict = json.load(f)
    sent_descr_audio_dict = get_sent_audio(conf.sent_topic_descr_embed_dir)

    # Reading the metadata.
    metadata_testset, topics_df, topics_df_targets = read_metadata_subset(conf, traintest='test')

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

    # Rename model paths. 
    model_path = Path(args.model_weights_path).parents[1]
    model_weights_path = args.model_weights_path
    model_config_path = os.path.join(model_path, 'config.json')

    # Opening JSON file
    f = open(model_config_path)
    model_config = json.load(f)
    CFG = from_dict(data_class=Cfg, data=model_config)
    CFG.device = "cuda" if torch.cuda.is_available() else "cpu"
    CFG.sp_path = conf.sp_path

    # Create dataloader of test set.
    data_loader = MMloader(CFG)

    # Load the model.
    full_model = mmModule(CFG)
    full_model.load_state_dict(torch.load(model_weights_path,  map_location=CFG.device)) 
    if args.create_random:
        # In case we want to create a random baseline.
        full_model = randomize_model(full_model) 

    full_model = full_model.to(CFG.device)     
    full_model.eval()

    # Calculate embeddings for test set.
    evaluator = Evaluator(CFG, model_path, full_model, data_loader, args.calc_acc)
    max_samples = evaluator.get_max_data()

    evaluator.encode_testset(max_samples) 
    evaluator.encode_queries(topics_df, query_field='query')
    evaluator.encode_queries(topics_df, query_field='description')
    evaluator.encode_sent_descr(sent_dict=sent_descr_dict, audio_dict=sent_descr_audio_dict)
    evaluator.add_query_labels(topics_df, val_df)

    # Create ranking and return results. 
    results = topic_evaluation(evaluator)

    # Save the results.
    save_eval_results(evaluator, results)


if __name__ == "__main__":
    # Parse flags in command line arguments
    parser = ArgumentParser()

    parser.add_argument('--model_weights_path', type=str, default="logs/test_model/output/full_model_weights.pt",
                        help='Folder where model weights are saved.')
    parser.add_argument('--calc_acc', action='store_true', default=False,
                    help='Whether to output accuracy.')
    parser.add_argument('--create_random', action='store_true', default=False,
                    help='If set to true, the model is initialized using random weights.')
    args, unparsed = parser.parse_known_args()

    main(args)
