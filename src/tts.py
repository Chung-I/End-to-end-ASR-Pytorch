import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.module import Encoder, Decoder, Postnet, CBHG, EncoderTaco, DecoderTaco, Linear, Highway


class Tacotron2(nn.Module):
    """Tacotron2 text-to-speech model (w/o stop prediction)
    """
    def __init__(self, n_mels, input_dim, paras):
    #def __init__(self, n_mels, linear_dim, in_embed_dim, enc_n_conv, enc_kernel_size, 
    #    enc_embed_dim, enc_dropout, n_frames_per_step, prenet_dim, prenet_dropout,
    #    query_rnn_dim, dec_rnn_dim, query_dropout, dec_dropout, 
    #    attn_dim, n_location_filters, location_kernel_size, loc_aware, use_summed_weights):
        super(Tacotron2, self).__init__()

        self.n_mels = n_mels
        self.loc_aware = paras['decoder']['loc_aware']
        self.use_summed_weights = paras['decoder']['use_summed_weights']

        #　self.encoder = Encoder(in_embed_dim, **paras['encoder'])
        self.decoder = Decoder(
            n_mels, enc_embed_dim=input_dim, **paras['decoder'])
        self.postnet = Postnet(n_mels, **paras['postnet'])
        self.n_frames_per_step = paras['decoder']['n_frames_per_step']

    def forward(self, inputs, input_lengths, teacher, tf_rate=0.0):
        """
        Arg:
            txt_embed: the output of TextEmbedding of shape (B, L, enc_embed_dim)
            txt_lengths: text lengths before padding (B)
            teacher: max_dec_step for inference. (B, T, n_mels) for training
            output_lengths: None for inference. (B) for training
            max_dec_steps: None for training. A python integer for inference
        """
        # enc_output = self.encoder(txt_embed, txt_lengths)
        mel, alignment, stop = self.decoder(inputs, input_lengths, teacher, tf_rate=tf_rate)
        mel_post = mel + self.postnet(mel)
        mel_pred = torch.stack([mel, mel_post], dim=1)
        return mel_pred, alignment, stop

    def create_msg(self):
        msg = []
        msg.append('Model spec.| Model = TACO - 2\t| Loc. aware = {}\t| Summed attn weights = {}\t| frames/step = {}\t'\
                   .format(self.loc_aware, self.use_summed_weights, self.decoder.n_frames_per_step))
        return msg



class Tacotron(nn.Module):
    def __init__(self, mel_dim, linear_dim, in_dim, paras):
        super(Tacotron, self).__init__()
        self.in_dim = in_dim
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        # self.encoder = EncoderTaco(in_dim, paras['encoder'])
        self.decoder = DecoderTaco(mel_dim, **paras['decoder'])
        self.postnet = CBHG(mel_dim, K=8, hidden_sizes=[256, mel_dim])
        self.last_linear = nn.Linear(mel_dim * 2, linear_dim)
        self.n_frames_per_step = paras['decoder']['n_frames_per_step']

    def forward(self, enc_outputs, enc_lengths, teacher, tf_rate=0.0):
        B = enc_outputs.size(0)

        # (B, T', mel_dim*r)
        mel_outputs, alignments = self.decoder(enc_outputs, teacher, tf_rate)

        # Post net processing below
        # Reshape
        # (B, T, mel_dim)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, alignments, linear_outputs

    def create_msg(self):
        msg = []
        msg.append('Model spec.| Model = TACO-1\t| frames/step = {}\t'\
                   .format(self.n_frames_per_step))
        return msg

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
        outputs = outputs.reshape(-1, outputs.size(-2) * self.ratio, outputs.size(-1) // self.ratio)
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
        outputs = outputs.reshape(-1, outputs.size(-2) * self.ratio, outputs.size(-1) // self.ratio)
        outputs.unsqueeze(1)
        return outputs, None, None

