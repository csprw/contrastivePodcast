"""
Self-supervised Contrastive Learning from Podcast Audio and Transcripts.
Author: Casper Wortmann
"""
import torch
from torch import nn
from transformers import AutoModel, AutoConfig
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class TextEncoder(nn.Module):
    """ The text-encoder. """
    def __init__(self, CFG):
        """
        Input:
            CFG: configuration of the training (dict). 
        """
        super().__init__()
        model_name_or_path = CFG.text_model_name
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = None

        config = AutoConfig.from_pretrained(model_name_or_path)
        self.hidden_dim = config.dim

        self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.max_seq_length = CFG.text_max_length

    def forward(self, text_embeds):
        """ Forward pass through the text-encoder. """
        trans_features = {'input_ids': text_embeds['input_ids'], 'attention_mask': text_embeds['attention_mask']}
        output_tokens = self.auto_model(**trans_features, return_dict=False)[0]
        features = {'input_ids':text_embeds['input_ids'], 'attention_mask':text_embeds['attention_mask'],'token_embeddings': output_tokens}
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

class SequentialAudioModel(nn.Module):
    """ Model of audio projection head that contains rnn-based layer. """
    def __init__(self, CFG):
        """
        Input:
            CFG: Configuration of the model (dict). 
        """
        super(SequentialAudioModel, self).__init__()
        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = CFG.audio_hidden_dim
        self.layer_dim = CFG.audio_layer_dim
        self.direction = 1
        
        self.device = CFG.device
        self.audio_model = CFG.audio_proj_head

        # RNN layers
        if self.audio_model == 'rnn':
            self.seq_model = nn.RNN(CFG.audio_encoder_input, 
                    CFG.audio_hidden_dim, CFG.audio_layer_dim, 
                    batch_first=True, dropout=CFG.audio_dropout)
        elif self.audio_model == 'gru':
            self.seq_model = nn.GRU(
                    input_size=CFG.audio_encoder_input, 
                    hidden_size=CFG.audio_hidden_dim, num_layers=CFG.audio_layer_dim, 
                    batch_first=True, dropout=CFG.audio_dropout)
        elif self.audio_model == 'mlp':
            self.seq_model = nn.GRU(
                input_size=CFG.audio_encoder_input, 
                hidden_size=CFG.audio_hidden_dim, num_layers=CFG.audio_layer_dim, 
                batch_first=True, dropout=CFG.audio_dropout)
        elif self.audio_model == 'lstm':
            self.seq_model = nn.GRU(
                input_size=CFG.audio_encoder_input, 
                hidden_size=CFG.audio_hidden_dim, num_layers=CFG.audio_layer_dim, 
                batch_first=True, dropout=CFG.audio_dropout)

        if self.audio_model in ['rnn', 'gru', 'lstm']:
            self.non_seq_layers = nn.Sequential(
                nn.Linear(CFG.audio_hidden_dim, CFG.mutual_embedding_dim),
                nn.Dropout(),
                nn.LayerNorm(CFG.mutual_embedding_dim),
            )

        elif self.audio_model == 'mlp':
            self.non_seq_layers = nn.Sequential(
                nn.Linear(CFG.audio_hidden_dim, CFG.audio_hidden_dim),
                nn.GELU(),
                nn.Linear(CFG.audio_hidden_dim, CFG.audio_hidden_dim),
                nn.Dropout(),
                nn.LayerNorm(CFG.mutual_embedding_dim),
            )

        self.direction = 2 if CFG.bidirectional else 1

    def forward(self, audio_seq):
        """ 
        Forward pass through the audio projection head. 
        Input: 
            audio_seq: tuple of yamnet embeddings and sentence lengths. 
        Output:
            embeds: the output of the forward pass (audio embeddings)
        """
        features, length = audio_seq

        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim * self.direction, features.size(0), self.hidden_dim).requires_grad_().to(self.device)

        if length != None:
            # Pack the features such that we do not compute zero products
            features = pack_padded_sequence(features, length, batch_first=True, enforce_sorted=False)
            out, h0 = self.seq_model(features, h0)
            out, _ = pad_packed_sequence(out, batch_first=True)
            out = h0[-1, :, :]
            
        else:
            # Forward propagation by passing in the input and hidden state into the model
            out, h0 = self.seq_model(features)   
            out = out[:, -1, :]  
        
        if self.audio_model in ['rnn', 'gru', 'lstm', 'mlp']:
            embeds = self.non_seq_layers(out)
            
        return embeds 

class simple_ProjectionHead(nn.Module):
    """
    Simple projection head for audio.
    """
    def __init__(self, CFG):
        super(simple_ProjectionHead, self).__init__()
        """
        Inputs:
            CFG: Configuration of the model (dict). 
        """
        self.input_dim = CFG.audio_encoder_input
        self.hidden_dim = CFG.audio_hidden_dim
        self.output_dim = CFG.final_projection_dim
        self.activation  = CFG.audio_activation
        dropout = CFG.audio_dropout

        self.config_keys = ['input_dim', 'hidden_dim', 'output_dim', 'activation']

        if self.activation == 'relu':
            self.simple_model = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.BatchNorm1d(num_features=self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        else:
            self.simple_model = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(self.output_dim),
            )

    def forward(self, feat_len):
        """ Forward pass through the projection head. """
        features, _ = feat_len
        features = self.simple_model(features)
        return features

class simple_ProjectionHead_text(nn.Module):
    """
    Simple projectionhead that can be used on top of the text encoder. 
    """
    def __init__(self, CFG):
        super().__init__()
        embedding_dim=CFG.mutual_embedding_dim
        projection_dim=CFG.final_projection_dim
        dropout=CFG.text_dropout
        self.activation = CFG.text_activation

        if self.activation == 'relu':
            self.simple_model = nn.Sequential(
                nn.Linear(embedding_dim, projection_dim),
                nn.BatchNorm1d(num_features=projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim),
            )
        else:
            self.simple_model = nn.Sequential(
                nn.Linear(embedding_dim, projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(projection_dim),
            )
    def forward(self, x):
        """ Forward pass through the projection head. """
        output_tokens = self.simple_model(x['token_embeddings'])
        features = {'input_ids':x['input_ids'], 'attention_mask':x['attention_mask'],'token_embeddings': output_tokens}
        return features


class Pooling(nn.Module):
    """
    Performs pooling on the token embeddings.
    """
    def __init__(self, pooling_mode='cls'):
        """
        Inputs:
            pooling_mode: poling strategy to use. 
        """
        super(Pooling, self).__init__()
        assert pooling_mode in [ 'max', 'cls', 'mean']
        self.pooling_mode = pooling_mode

    def forward(self, features):
        """ Pooling forward pass. """
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']

        # Pooling strategy
        if self.pooling_mode == 'cls':
            cls_token = features.get('cls_token_embeddings', token_embeddings[:, 0])  # Take first token by default
            features.update({'sentence_embedding': cls_token})

        if self.pooling_mode == 'max':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            features.update({'sentence_embedding': max_over_time})

        if self.pooling_mode == 'mean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pool = sum_embeddings / sum_mask
            features.update({'sentence_embedding': mean_pool})

        return features
