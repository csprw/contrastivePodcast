"""
Self-supervised Contrastive Learning from Podcast Audio and Transcripts.
Author: Casper Wortmann
Usage: python train.py --args
"""
import sys
from time import time
from argparse import ArgumentParser
import logging
import csv
import os
import math
from collections import Counter, OrderedDict

from dataclasses import dataclass
from omegaconf import OmegaConf
import pathlib

import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_constant_schedule, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import logging

from utils import setup_config, set_logconfig, print_best_results, get_metrics, to_plot
from models import TextEncoder, SequentialAudioModel, simple_ProjectionHead, simple_ProjectionHead_text, Pooling
from dataloader import MMloader 
logging.set_verbosity_error()

class mmModule(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.batch_size = CFG.batch_size
        self.device = CFG.device
        self.loss_type = CFG.loss_type

        if CFG.text_proj_head == 'sph':
            text_encoder = TextEncoder(CFG)
            text_projection = simple_ProjectionHead_text(CFG)
            pooling_model = Pooling(CFG.text_pooling)
            text_modules = [text_encoder, text_projection, pooling_model]

        elif CFG.text_proj_head.lower() == 'none':
            text_encoder = TextEncoder(CFG)
            pooling_model = Pooling(CFG.text_pooling)
            text_modules = [text_encoder, pooling_model]

        if CFG.audio_proj_head in ['rnn', 'gru', 'mlp', 'lstm']:
            audio_modules = [SequentialAudioModel(CFG)]
        elif CFG.audio_proj_head in ['sph']:
            audio_modules = [simple_ProjectionHead(CFG)]
        
        # Create the full model
        if text_modules is not None and not isinstance(text_modules, OrderedDict):
            text_modules = OrderedDict([(str(idx), module) for idx, module in enumerate(text_modules)])
        if audio_modules is not None and not isinstance(audio_modules, OrderedDict):
            audio_modules = OrderedDict([(str(idx), module) for idx, module in enumerate(audio_modules)])

        self.text_model = nn.Sequential(text_modules)
        self.audio_model = nn.Sequential(audio_modules)

        self.eval_every = CFG.eval_every  
        self.print_every = CFG.print_every
        self.batch_size = CFG.batch_size
        self.device = CFG.device

        self.log_name = CFG.log_name
        self.init_logging()
        self.best_loss = float('inf')

    def _get_scheduler(self, optimizer, scheduler: str, warmup_steps: int):
        """
        Returns the a scheduler for the learning rate. 
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=self.num_train_steps)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def _get_optimizer(self, loss_model):
        """
        Returns the optimizer. 
        """
        parameters = list(loss_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer_params = self.optimizer_params
        optimizer = self.optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        return optimizer

    def fit(self,
            CFG,
            train_loader,
            val_loader,
            steps_per_epoch,
            loss_model,
            start_epoch=1,
            optimizer_class=AdamW,
            loaded_optimizer_state = None,
            loaded_sched_state = None
        ):
        # Push to GPU or cpu.
        self.text_model.to(self.device)
        self.audio_model.to(self.device)
        loss_model.to(self.device)

        self.val_loader = val_loader
        warmup_steps =  math.ceil(steps_per_epoch *  CFG.num_epochs * 0.1)  
        self.weight_decay = CFG.weight_decay
        self.optimizer_class = optimizer_class
        self.optimizer_params = {'lr': CFG.lr}
        scheduler_method = 'WarmupLinear' 
        self.num_train_steps = int(steps_per_epoch * CFG.num_epochs)

        # If we continue a paused run, write results at the right step.
        steps_so_far = (start_epoch + 1)

        # Initiate or load an optimizer.
        optimizer = self._get_optimizer(loss_model)
        if loaded_optimizer_state != None:
            optimizer = self._get_optimizer(loss_model)
            optimizer.load_state_dict(loaded_optimizer_state)

        # Initiate or load a scheduler.
        scheduler = self._get_scheduler(optimizer, scheduler=scheduler_method, warmup_steps=warmup_steps)
        if loaded_sched_state != None:
            scheduler.load_state_dict(loaded_sched_state)

        for epoch in range(start_epoch, CFG.num_epochs):
            t1 = time()
            loss_model.zero_grad()
            loss_model.train()
            for step, batch in enumerate(iter(train_loader)):

                if  step % self.eval_every == 0 or step == steps_per_epoch - 1: 
                    # Evaluate on validation set. 
                    mean_loss, metrics = self.evaluate(loss_model)
                    self.add_logging(epoch, steps_so_far, mean_loss, metrics, train=False)
                    
                    print("[Eval] Epoch {} Step {}/{} \t loss {} \t acc {}".format(epoch, step, self.num_train_steps, mean_loss, metrics['mean_acc']))
                    if mean_loss < self.best_loss:
                        print("[Eval] better model found")
                        self.best_loss = mean_loss 
                        if args.save_model:
                            self.save_model()
                
                    # If csv files are properly set up, save plots.
                    if os.path.isfile(self.train_csv_filename):
                        self.output_all_plots()

                sent_features, audio_features, seq_len = batch
                loss_value, metrics = loss_model(sent_features, audio_features, seq_len)

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                scheduler.step()
                steps_so_far += 1

                if step % self.print_every == 0:
                    print("[Fit] Epoch {} Step {}/{} \t loss {} \t acc {}".format(epoch, step, num_train_steps, loss_value.item(), metrics['mean_acc']))
                    self.add_logging(epoch, steps_so_far, loss_value.item(), metrics, train=True)

            self.output_all_plots()
            t2 = time()
            print("[Fit] epoch duration {} seconds".format(int(t2-t1)))
            if args.save_model:
                self.save_model("_epoch_"+str(epoch))
            if args.save_checkpoint:
                self.save_checkpoint(epoch, step, optimizer, scheduler)

    def evaluate(self, loss_model):
        loss_model.eval()
        losses = 0
        self.to(self.device)

        iterator = iter(self.val_loader)
        total_len = 1000   # for now I evaluate on a subset

        with torch.no_grad():
            for step in range(total_len):
                batch = next(iterator)
                sent_features, audio_features, seq_len  = batch
                with torch.no_grad():
                    loss_value, metrics = loss_model(sent_features, audio_features, seq_len)
                    losses += loss_value.detach().cpu().item()

                    if step == 0:
                        met_sum = Counter(metrics.copy())
                    else:
                        met_sum.update(Counter(metrics))

        mean_metrics = {k: value / total_len  for k, value in met_sum.items()}
        mean_loss = losses / total_len

        del met_sum
        loss_model.train()
        return mean_loss, mean_metrics

    def output_all_plots(self):
        to_plot(self.train_csv_filename, column='audio_acc', title="Train accuracy (audio)")
        to_plot(self.train_csv_filename, column='text_acc', title="Train accuracy (text)")
        to_plot(self.train_csv_filename, column='loss', title="Train loss")
        to_plot(self.eval_csv_filename, column='mean_acc', title="val accuracy (mean)")
  
    def init_logging(self):
        self.model_save_path = '{}/output'.format(self.log_name)
        os.makedirs(self.model_save_path, exist_ok=True)
        self.train_csv_filename = os.path.join(self.model_save_path, "train.csv")
        self.eval_csv_filename = os.path.join(self.model_save_path, "val.csv")

    def init_csv(self, metrics):
        for filename in [self.train_csv_filename, self.eval_csv_filename]:
            self.metric_headers = [metric for metric in metrics.keys()]
            self.train_csv_headers = ["epoch", "steps", "loss"] + self.metric_headers
            with open(filename, newline='', mode='w', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.train_csv_headers)

    def add_logging(self, epoch, steps, loss, metrics, train=True):
        if train:
            filename = self.train_csv_filename
        else:
            filename = self.eval_csv_filename

        output_file_exists = os.path.isfile(filename)
        if not output_file_exists:
            self.init_csv(metrics)
            
        # Add metrics to CSV
        metric_vals = [metrics[header] for header in self.metric_headers]
        with open(filename, newline='', mode="a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, steps, loss] + metric_vals)

    def save_model(self, extra_name=""):
        # Save the model
        output_dir = os.path.join(self.model_save_path, '{}{}_weights.pt'.format('full_model', extra_name))
        torch.save(self.state_dict(), output_dir)

    def save_checkpoint(self, epoch, step, optimizer, scheduler):
        checkpoint = { 
            'epoch': epoch,
            'step': step,
            'full_model': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_sched': scheduler.state_dict()
        }
        output_dir = os.path.join(self.model_save_path, 'checkpoint.pt')
        torch.save(checkpoint, output_dir)

class multimodal_loss(nn.Module):
    """
    This loss expects as input a batch consisting of transcript and 
    audio encodings. 
    """
    def __init__(self, full_model, CFG):
        super(multimodal_loss, self).__init__()

        self.text_model = full_model.text_model
        self.audio_model = full_model.audio_model
        self.loss_type = CFG.loss_type
        self.scale_type = CFG.scale_type
        self.scale = CFG.scale

        if self.scale_type == 'fixed':
            self.fixed_scale = self.scale
        elif self.scale_type == 'learned':
            self.logit_scale = nn.Parameter(torch.log(torch.ones([]) * 100))
            self.logit_scale.requires_grad = True

        self.device = torch.device(CFG.device)
        self.batch_size = full_model.batch_size

        self.loss_audio = nn.CrossEntropyLoss()
        self.loss_text = nn.CrossEntropyLoss()

    def forward(self, sentence_features, audio_features, seq_len):
        # Get sentence Representations
        reps_text = self.text_model(sentence_features)['sentence_embedding']

        # Get Audio representations. 
        reps_audio = self.audio_model((audio_features, seq_len))

        # Normalize representations.
        reps_audio = torch.nn.functional.normalize(reps_audio) 
        reps_text = torch.nn.functional.normalize(reps_text) 
                
        # Calulate logits.
        if self.scale_type == 'fixed':
            audio_logits =  (reps_audio @ reps_text.t()) * self.fixed_scale
        elif self.scale_type == 'learned':
            cur_logit_scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
            audio_logits =  (reps_audio @ reps_text.t()) * cur_logit_scale.exp()
        text_logits = audio_logits.t()

        # Calculate metrics. 
        ground_truth = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
        total_loss =  (self.loss_text(text_logits, ground_truth) + self.loss_audio(audio_logits,ground_truth))/2
        metrics = get_metrics(audio_logits.detach(), text_logits.detach(), ground_truth)
        return total_loss, metrics

    def get_config_dict(self):
        return {
            'scale_type': self.scale_type, 
            'similarity_fct': self.similarity_fct.__name__
        }


@dataclass
class Cfg:
    batch_size: int = 128
    num_epochs: int = 1
    loss_type: str = 'simcse_loss'
    lr: float = 5e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # For the audio module
    audio_encoder_input: int = 1024
    audio_hidden_dim: int = 768
    audio_layer_dim: int = 2
    audio_activation: str = 'relu'

    # For the text module
    text_model_name : str = 'distilbert-base-uncased'
    text_tokenizer: str = "distilbert-base-uncased"
    text_pooling: str = 'cls'   #['cls', 'max']
    text_max_length: int = 32

    # For the projection_head modules
    text_proj_head: str = 'None'
    audio_proj_head: str = 'None'
    text_activation: str = 'gelu'

    final_projection_dim: int = 768  # [256 or 768]
    mutual_embedding_dim: int = 768
    audio_dropout: float = 0.1
    text_dropout: float = 0.1
    weight_decay: float = 0.01

    text_activation: str = ''
    audio_activation: str = ''
    scale_type: str = ''
    scale: int = 20
    bidirectional: bool = False
    eval_every: int = 1
    print_every: int = 1
    log_name: str = ''

    train_dataset: str = ''
    val_dataset: str = ''
    test_dataset: str = ''
    seed: int = 100

    max_train_samples: int = 0
    save_model: bool = False
    save_checkpoint: bool = False
    load_model_path: str = ''
    load_model: bool = False
    load_checkpoint: bool = False
    load_checkpoint_path: str = ''
    weak_shuffle: bool = False


def main(args, conf):
    # Setup configuration.
    t_start = time()
    set_logconfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FullCfg = setup_config(args, Cfg, device)
    FullCfg.sp_path = conf.sp_path
    FullCfg.sp_sample_path = conf.sp_sample_path

    # Setup dataloaders.
    data_loader = MMloader(FullCfg)
    train_loader = data_loader.train_loader
    val_loader = data_loader.val_loader
    steps_per_epoch = int(len(data_loader.train_dataset) / FullCfg.batch_size)

    # Evaluate every 10% of the data, print result every 2%.
    FullCfg.eval_every = int(math.ceil(steps_per_epoch * 0.1)) 
    FullCfg.print_every = int(math.ceil(steps_per_epoch * 0.02))
    print("[main] print_every {} eval_every {} ".format(
        FullCfg.print_every, FullCfg.eval_every)
    )

    # Setup the model and loss function.
    full_model = mmModule(FullCfg)
    loss_func = multimodal_loss(full_model, FullCfg)

    if args.load_checkpoint:
        # Load a previous run to continue training. 
        print("[Main] Continue run {} ".format(args.load_checkpoint_path))
        
        # To continue a linux-run on Windows the following is needed.
        plt = sys.platform
        if plt != 'win32': 
            pathlib.WindowsPath = pathlib.PosixPath

        checkpoint = torch.load(args.load_checkpoint_path, map_location=torch.device(device))
        epoch = checkpoint['epoch'] + 1
        full_model.load_state_dict(checkpoint['full_model'])
        loaded_optimizer_state = checkpoint['optimizer']
        loaded_sched_state = checkpoint['lr_sched']
        full_model.device = FullCfg.device

    else:
         # Initiate a new run.
        if args.load_model: 
            print("[Main] load model {}, but initiate new optimizer".format( 
                args.load_model_path)
            )
            full_model.load_state_dict(torch.load(args.load_model_path))
        epoch = 0
        loaded_optimizer_state = None
        loaded_sched_state = None
        
    # Fit the model. 
    full_model.fit(
        CFG = FullCfg,
        train_loader=train_loader,
        val_loader = val_loader,
        steps_per_epoch = steps_per_epoch,
        loss_model=loss_func,
        start_epoch=epoch,
        optimizer_class=AdamW,
        loaded_optimizer_state = loaded_optimizer_state,
        loaded_sched_state = loaded_sched_state
    )

    t_end = time()
    dur = int(t_end - t_start)
    print("[Main] Done, total duration {} seconds ".format(dur))
    print_best_results(full_model.eval_csv_filename, dur, FullCfg.log_name)




if __name__ == "__main__":
    # Load static configurations.
    conf = OmegaConf.load("./config.yaml")

    # Parse flags in command line arguments.
    parser = ArgumentParser()

    parser.add_argument('--train_dataset', default='sp_sample', const='sp_sample',
                    nargs='?', choices=['sp_sample', 'sp'],
                    help='Name of training dataset (default: %(default)s)')
    parser.add_argument('--val_dataset', default='sp_sample', const='sp_sample',
                    nargs='?', choices=['sp_sample',  'sp'],
                    help='Name of validation dataset (default: %(default)s)')
    parser.add_argument('--test_dataset', default='sp_sample', const='sp_sample',
                    nargs='?', choices=['sp_sample',  'sp'],
                    help='Name of test dataset (default: %(default)s)')

    parser.add_argument('--seed', type=int, default=100,
                        help='Seet to use')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs to train')
    
    parser.add_argument('--log_dir', type=str, default="./logs",
                        help='Folder where so save logs.')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='batch_size')

    parser.add_argument('--final_projection_dim', type=int, default=768, 
                    nargs='?', choices=[256, 768],
                    help='Final output dimensions of the embeddings')
    parser.add_argument('--loss_type', default='clip_loss', const='clip_loss',
                    nargs='?', choices=['simcse_loss', 'clip_loss', 'norm_loss'],
                    help='Name of scale_type (default: %(default)s)')
    parser.add_argument('--audio_proj_head', default='gru', const='gru',
                    nargs='?', choices=['sph', 'rnn', 'gru', 'lstm', 'mlp'],
                    help='Activation to use in simple proj head (default: %(default)s)')
    parser.add_argument('--text_proj_head', default='None', const='None',
                    nargs='?', choices=['sph', 'None'],
                    help='Activation to use in simple proj head (default: %(default)s)')
    parser.add_argument('--text_pooling', default='mean', const='mean',
                    nargs='?', choices=['cls', 'mean', 'max'],
                    help='Pooling method to use for text model (default: %(default)s)')

    parser.add_argument('--audio_activation', default='relu', const='relu',
                    nargs='?', choices=['relu', 'gelu'],
                    help='Activation to use in simple proj head (default: %(default)s)')
    parser.add_argument('--text_activation', default='gelu', const='gelu',
                    nargs='?', choices=['relu', 'gelu'],
                    help='Activation to use in simple proj head (default: %(default)s)')
    parser.add_argument('--scale_type', default='fixed', const='fixed',
                    nargs='?', choices=['fixed', 'learned'],
                    help='Name of scale_type (default: %(default)s)')
    parser.add_argument('--scale', type=float, default=20,
                        help='Fixed scale to use')

    parser.add_argument('--save_model', dest='save_model', action='store_true',
                        help="Save the model weights.")
    parser.add_argument('--load_model_path', type=str, default="./logs/logname/output/full_model_weights.pt",
                        help='Folder where model weights are saved.')
    parser.add_argument('--load_model', dest='load_model', action='store_true',
                        help="Load the model in model_path to continue a downstream task")

    parser.add_argument('--save_checkpoint', dest='save_checkpoint', action='store_true',
                        help="Save the model, optimizer and scheduler weights.")
    parser.add_argument('--load_checkpoint_path', type=str, default="./logs/logname/output/checkpoint.pt",
                        help='Folder where so save logs.')
    parser.add_argument('--load_checkpoint', dest='load_checkpoint', action='store_true',
                        help="Load a model, optimizer and scheduler to continue training.")

    parser.add_argument('--max_train_samples', type=int, default=0,
                        help='Maximum number of training samples used.')
    args, unparsed = parser.parse_known_args()

    main(args, conf)