from functools import partial
import os
import math

from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.solver import BaseSolver

from src.asr import ASR
from src.tts import FeedForwardTTS, HighwayTTS, Tacotron, Tacotron2
from src.id_net import RNNSimple
from src.netvlad import ThinResNet
from src.optim import Optimizer
from src.data import load_dataset
from src.module import RNNLayer
from src.util import human_format, cal_er, feat_to_fig, freq_loss, \
    get_mask_from_sequence_lengths, get_grad_norm, roc_score, cm_figure

DEV_N_EXAMPLES = 0  # How many examples to show in tensorboard
CKPT_STEP = 10000
CKPT_EPOCH = 10


class Solver(BaseSolver):
    ''' Solver for testing id_net (EER scoring)'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        # Logger settings
        self.best_eer = 3.0
        self.best_tts_loss = float('inf')
        # Curriculum learning affects data loader
        self.curriculum = self.config['hparas']['curriculum']

    def fetch_data(self, data):
        ''' Move data to device and compute text seq. length'''
        _, feat, feat_len, txt, (spkr_id, dataset_idx, to_save) = data
        feat = feat.to(self.device)
        feat_len = feat_len.to(self.device)
        txt = txt.to(self.device)
        txt_len = torch.sum(txt != 0, dim=-1)
        spkr_id = spkr_id.to(self.device)

        return feat, feat_len, txt, txt_len, (spkr_id, dataset_idx, to_save)

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        _, self.dv_set, self.tokenizer, self.audio_converter, msg, (self.spkr_weight, self.spkr_id_list) = load_dataset(
            self.paras.njobs, self.paras.gpu, self.paras.pin_memory, self.curriculum > 0, **self.config['data'])
        self.vocab_size = self.tokenizer.vocab_size
        self.feat_dim, _ = self.audio_converter.feat_dim                  # ignore linear dim
        self.spkr_num = len(self.spkr_weight)
        self.spkr_weight = self.spkr_weight.to(self.device)
        self.verbose(msg)

    def set_model(self):
        ''' Setup ASR model and optimizer '''
        # Model
        id_in_dim = 80

        if self.config['id_net']['type'] == "netvlad":
            self.config['id_net'].pop('type')
            self.id_net = ThinResNet(
                self.spkr_num, **self.config['id_net']).to(self.device)
        else:
            self.config['id_net'].pop('type')
            self.id_net = RNNSimple(
                id_in_dim, self.spkr_num, **self.config['id_net']).to(self.device)

        # self.verbose(self.asr.create_msg())
        model_paras = [{'params': self.id_net.parameters()}]

        # Optimizer
        self.optimizer = Optimizer(model_paras, **self.config['hparas'])
        self.verbose(self.optimizer.create_msg())

        # Enable AMP if needed
        self.enable_apex()

        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt_dir()

    def load_ckpt_dir(self):
        if self.paras.load:
            ckpt_dir = Path(self.paras.load)
            self.ckpt_list = list(ckpt_dir.rglob("*.pth"))
        else:
            raise NotImplementedError


    def load_ckpt(self, ckpt_file):
        ''' Load ckpt if --load option is specified '''
        # Load weights
        #ckpt = torch.load(ckpt_file, map_location=self.device if self.mode == 'train' else 'cpu')
        ckpt = torch.load(ckpt_file, map_location=self.device)

        # Load task-dependent items
        if self.mode == 'train':
            raise NotImplementedError
        else:
            self.id_net.load_state_dict(ckpt['id_net'])
            self.step = ckpt['global_step']
            for k, v in ckpt.items():
                if type(v) is float:
                    metric, score = k, v
            self.verbose('Evaluation target = {} (recorded {} = {:.2f} %)'.format(
                str(ckpt_file), metric, score))

    def exec(self):
        ''' Testing identification neural network on the training split'''
        n_epochs = 0
        dev_spkr_id = []
        for ckpt_file in self.ckpt_list:
            self.timer.set()
            self.load_ckpt(ckpt_file)
            self.id_net.eval()

            dev_spkr_emb = []
            for i, data in enumerate(self.dv_set):
                self.progress('Valid step - {}/{}'.format(i+1, len(self.dv_set)))

                # Fetch data
                feat, feat_len, _, _, (spkr_id, dataset_idx, to_save) = self.fetch_data(data)
                self.timer.cnt('rd')

                if n_epochs == 0:
                    dev_spkr_id += spkr_id.cpu().tolist()
                    for index, saved_feat in zip(dataset_idx, to_save):
                        self.tr_set.dataset.set_feat(index, saved_feat)

                # Forward model
                with torch.no_grad():
                    spkr_embedding, _ = self.id_net(feat)
                    dev_spkr_emb += spkr_embedding.cpu().tolist()

                torch.cuda.empty_cache()

            assert len(dev_spkr_emb) == len(dev_spkr_id)
            eer, dcf2, dcf3 = roc_score(dev_spkr_emb, dev_spkr_id)
            self.write_log('ROC', {'EER': eer, 'DCF2': dcf2, 'DCF3': dcf3})

        self.log.close()

