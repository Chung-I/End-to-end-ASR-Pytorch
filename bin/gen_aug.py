from functools import partial
import os
import math
from pathlib import Path
from tqdm import tqdm
from itertools import chain

import torch
import torch.nn as nn
from src.solver import BaseSolver
import torchaudio

from src.asr import ASR
from src.tts import FeedForwardTTS, HighwayTTS, Tacotron, Tacotron2
from src.optim import Optimizer
from src.data import load_dataset
from src.module import RNNLayer
from src.util import human_format, cal_er, feat_to_fig, freq_loss, \
    get_mask_from_sequence_lengths, get_grad_norm

DEV_N_EXAMPLES = 16  # How many examples to show in tensorboard
CKPT_STEP = 10000


class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        # Logger settings
        assert self.config['data']['corpus']['name'] == self.src_config['data']['corpus']['name']
        self.config['data']['corpus']['path'] = self.src_config['data']['corpus']['path']
        self.config['data']['corpus']['bucketing'] = False

        # The follow attribute should be identical to training config
        #self.config['data']['audio'] = self.src_config['data']['audio']
        self.config['data']['text'] = self.src_config['data']['text']
        self.config['model'] = self.src_config['model']
        self.config['tts'] = self.src_config['tts']

    def fetch_data(self, data):
        ''' Move data to device and compute text seq. length'''
        file, feat, feat_len, txt, spkr_id = data
        if hasattr(self.tts, 'n_frames_per_step'):
            bs, timesteps, _ = feat.size()
            padded_timesteps = timesteps + self.tts.n_frames_per_step - \
                (timesteps % self.tts.n_frames_per_step)
            padded_feat = feat.new_zeros((bs, padded_timesteps, self.feat_dim))
            padded_feat[:, :timesteps, :] = feat
            feat = padded_feat
        feat = feat.to(self.device)
        feat_len = feat_len.to(self.device)
        txt = txt.to(self.device)
        txt_len = torch.sum(txt != 0, dim=-1)

        return file, feat, feat_len, txt, txt_len

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.out_path = Path(os.path.join(self.paras.outdir, self.paras.name))
        self.path = Path(self.config['data']['corpus']['path'])
        self.dv_set, self.tt_set, self.tokenizer, self.audio_converter, msg, _ = \
            load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory,
                         False, **self.config['data'], task='tts')
        self.vocab_size = self.tokenizer.vocab_size
        self.feat_dim, _ = self.audio_converter.feat_dim                  # ignore linear dim
        self.verbose(msg)

    def set_model(self):
        ''' Setup ASR model and optimizer '''
        # Model
        self.asr = ASR(self.feat_dim, self.vocab_size, **
                       self.config['model']).to(self.device)
        self.layer_num = self.config['tts']['layer_num']
        with torch.no_grad():
            seq_len = 64
            n_channels = self.config['model']['delta'] + 1
            dummy_inputs = torch.randn(
                (1, seq_len, n_channels * self.feat_dim)).to(self.device)
            dummy_feat_len = torch.full((1, ), seq_len)
            dummy_outs, dummy_out_len, _ = \
                self.asr.encoder.get_hidden_states(
                    dummy_inputs, seq_len, self.layer_num)
            tts_upsample_rate = (dummy_feat_len / dummy_out_len).int().item()
            tts_in_dim = dummy_outs.size(-1)

        if self.config['tts']['type'] == "linear":
            # self.asr.encoder.layers[self.layer_num].out_dim
            self.tts = FeedForwardTTS(tts_in_dim,
                                      self.feat_dim, self.config['tts']['num_layers'],
                                      tts_upsample_rate).to(self.device)
        elif self.config['tts']['type'] == "highway":
            self.tts = HighwayTTS(tts_in_dim,
                                  self.feat_dim, self.config['tts']['num_layers'],
                                  tts_upsample_rate).to(self.device)
        elif self.config['tts']['type'] == "tacotron2":
            self.tts = Tacotron2(self.feat_dim,
                                 tts_in_dim, self.config['tts']).to(self.device)
        else:
            raise NotImplementedError

        self.verbose(self.asr.create_msg())
        # self.verbose(self.tts.create_msg())
        model_paras = [{'params': self.asr.parameters()},
                       {'params': self.tts.parameters()}]
        for param in self.asr.parameters():
            param.requires_grad = False

        # Enable AMP if needed
        self.enable_apex()

        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt(cont=False)

    def load_ckpt(self, cont=True):
        ''' Load ckpt if --load option is specified '''
        # Load weights
        ckpt = torch.load(self.paras.load, map_location=self.device)
        self.asr.load_state_dict(ckpt['asr'])
        self.tts.load_state_dict(ckpt['tts'])

    def exec(self):
        ''' Training End-to-end ASR system '''

        self.asr.eval() # behavior of generating spectrogram should be in eval mode
        self.tts.eval()

        for data in tqdm(chain(self.dv_set, self.tt_set)):
            # Fetch data
            continue

