import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.module import Encoder, Decoder, Postnet, CBHG, EncoderTaco, DecoderTaco, Linear, Highway


class RNNSimple(nn.Module):
    """docstring for RNNSimple"""

    def __init__(self, input_dim, spkr_num, spkr_dim, loss_fn):
        super(RNNSimple, self).__init__()
        self.embedder = nn.LSTM(input_size=input_dim,
                                hidden_size=128, num_layers=3, batch_first=True)
        self.linear = nn.Linear(in_features=128, out_features=spkr_dim)
        self.relu = nn.ReLU()
        self.pred_layer = nn.Linear(spkr_dim, spkr_num, bias=False)
        self.loss_fn = loss_fn

    def forward(self, input_feat, input_len, hidden_init=None):
        B = input_feat.size(0)
        input_feat = input_feat.squeeze()
        input_feat_packed = pack_padded_sequence(
            input_feat, input_len, batch_first=True)
        _, (hidden, _) = self.embedder(input_feat_packed, hidden_init)
        spkr_embedding = self.linear(hidden[-1])
        #spkr_embedding = self.relu(spkr_embedding)
        if self.loss_fn == 'softmax':
            spkr_pred = self.pred_layer(spkr_embedding)
            spkr_pred = F.softmax(spkr_pred, dim=-1)
        elif self.loss_fn == 'amsoftmax':
            spkr_embedding = F.normalize(spkr_embedding, p=2, dim=-1)
            spkr_pred = self.pred_layer(spkr_embedding)
        else:
            raise NotImplementedError

        spkr_embedding = spkr_embedding.view(B, -1)
        spkr_pred = spkr_pred.view(B, -1)
        return spkr_embedding, spkr_pred


class TAPSimple(nn.Module):
    def __init__(self, input_dim, spkr_num, spkr_dim, loss_fn):
        super(TAPSimple, self).__init__()
        self.embedder = nn.Linear(input_dim, spkr_dim)
        self.pred_layer = nn.Linear(spkr_dim, spkr_num, bias=False)
        self.loss_fn = loss_fn

    def forward(self, input_feat):
        B = input_feat.size(0)
        tap_feat = input_feat.mean(dim=-2)
        spkr_embedding = self.embedder(tap_feat)
        if self.loss_fn == 'softmax':
            spkr_pred = self.pred_layer(spkr_embedding)
            spkr_pred = F.softmax(spkr_pred, dim=-1)
        elif self.loss_fn == 'amsoftmax':
            spkr_embedding = F.normalize(spkr_embedding, p=2, dim=-1)
            spkr_pred = self.pred_layer(spkr_embedding)
        else:
            raise NotImplementedError

        spkr_embedding = spkr_embedding.view(B, -1)
        spkr_pred = spkr_pred.view(B, -1)
        return spkr_embedding, spkr_pred


class FeedForwardTTS(nn.Module):
    def __init__(self, input_dim, mel_dim, num_layers, ratio):
        super(FeedForwardTTS, self).__init__()
        self.ratio = ratio
        hidden_dim = mel_dim * ratio
        self.decoder = nn.Sequential(
            *[nn.Linear(input_dim, hidden_dim), nn.Tanh()],
            *((num_layers - 1) * [nn.Linear(hidden_dim, hidden_dim),
                                  nn.Tanh()]),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, inputs, input_lengths, teacher, tf_rate=0.0):
        outputs = self.decoder(inputs)
        outputs = outputs.reshape(-1, outputs.size(-2)
                                  * self.ratio, outputs.size(-1) // self.ratio)
        outputs.unsqueeze(1)
        return outputs, None, None


class HighwayTTS(nn.Module):
    def __init__(self, input_dim, mel_dim, num_layers, ratio):
        super(HighwayTTS, self).__init__()
        self.ratio = ratio
        hidden_dim = mel_dim * ratio
        self.decoder = nn.Sequential(
            Linear(input_dim, hidden_dim),
            *((num_layers - 1) * [Highway(hidden_dim, hidden_dim)]),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, inputs, input_lengths, teacher, tf_rate=0.0):
        outputs = self.decoder(inputs)
        outputs = outputs.reshape(-1, outputs.size(-2)
                                  * self.ratio, outputs.size(-1) // self.ratio)
        outputs.unsqueeze(1)
        return outputs, None, None
