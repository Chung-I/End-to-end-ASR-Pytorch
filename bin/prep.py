import torch
from src.solver import BaseSolver

from src.asr import ASR
from src.optim import Optimizer
from src.data import load_dataset
from src.util import human_format, cal_er, feat_to_fig

class Solver(BaseSolver):
    ''' Solver for training'''
    def __init__(self,config,paras,mode):
        super().__init__(config,paras,mode)
        # Logger settings
        self.best_wer = {'att':3.0,'ctc':3.0}
        # Curriculum learning affects data loader
        self.curriculum = self.config['hparas']['curriculum']

    def fetch_data(self, data):
        ''' Move data to device and compute text seq. length'''
        _, feat, feat_len, txt, spkr_id = data
        feat = feat.to(self.device)
        feat_len = feat_len.to(self.device)
        txt = txt.to(self.device)
        txt_len = torch.sum(txt!=0,dim=-1)
        
        return feat, feat_len, txt, txt_len

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        # self.config['data']['corpus']['save'] = True
        self.tr_set, self.dv_set, self.tokenizer, self.audio_converter, msg = \
                         load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory, 
                                      self.curriculum>0, **self.config['data'])
        self.vocab_size = self.tokenizer.vocab_size
        self.feat_dim, _ = self.audio_converter.feat_dim                  # ignore linear dim   
        self.verbose(msg)

    def set_model(self):
        pass


    def exec(self):
        pass
