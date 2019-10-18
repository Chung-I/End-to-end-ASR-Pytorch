import copy
import torch
import tensorflow as tf
from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed

from src.solver import BaseSolver
from src.asr import ASR
from src.decode import BeamDecoder
from src.data import load_dataset
from src.util import cal_er

BEAM_WIDTH = 100
TOP_PATHS = 20

class Solver(BaseSolver):
    ''' Solver for training'''
    def __init__(self,config,paras,mode):
        super().__init__(config,paras,mode)
        
        # ToDo : support tr/eval on different corpus
        assert self.config['data']['corpus']['name'] == self.src_config['data']['corpus']['name']
        #self.config['data']['corpus']['path'] = self.src_config['data']['corpus']['path']
        self.config['data']['corpus']['bucketing'] = False

        # The follow attribute should be identical to training config
        #self.config['data']['audio'] = self.src_config['data']['audio']
        self.config['data']['text'] = self.src_config['data']['text']
        self.config['model'] = self.src_config['model']

        # Output file
        self.output_file = str(self.ckpdir)+'_{}_{}.csv'

        # Override batch size for beam decoding
        self.beam_size = self.config['decode']['beam_size']
        self.greedy = self.config['decode']['beam_size'] == 1
        # if not self.greedy:
        #     self.config['data']['corpus']['batch_size'] = 1
        # else:
        #     # ToDo : implement greedy
        #     raise NotImplementedError

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
        self.dv_set, self.tt_set, self.tokenizer, self.audio_converter, msg, _ = \
                         load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory, 
                                      False, **self.config['data'], batch_for_dev=True)
        self.vocab_size = self.tokenizer.vocab_size
        self.feat_dim, _ = self.audio_converter.feat_dim                  # ignore linear dim

    def set_model(self):
        ''' Setup ASR model '''
        # Model
        self.model = ASR(self.feat_dim, self.vocab_size, **self.config['model']).to(self.device)
        # Load target model in eval mode
        self.load_ckpt()

    def exec(self):
        ''' Testing End-to-end ASR system '''
        for subset, ds in zip(['dev','test'],[self.dv_set, self.tt_set]):
            wers = []
            for data in tqdm(ds):
                feats, feat_lens, txts, txt_lens = self.fetch_data(data)
                with torch.no_grad():
                    logits, encode_lens, _, _, _ = \
                        self.model(feats, feat_lens, 0, log_prob=False)
                    if self.beam_size > 1:
                        logits = logits.transpose(1, 0)
                        timesteps, batch_size, _ = logits.size()
                        tf_tensor = tf.convert_to_tensor(logits.cpu().numpy(), dtype=tf.float32)
                        tf_slen = tf.convert_to_tensor(encode_lens.cpu().numpy(), dtype=tf.int32)
                        decoded, _ = tf.nn.ctc_beam_search_decoder(tf_tensor, tf_slen,
                                                                   beam_width=self.beam_size,
                                                                   top_paths=1)
                        pred = tf.sparse.to_dense(decoded[0]).numpy()
                        logits = pred

                wers += cal_er(self.tokenizer, logits, txts, ctc=True, return_list=True)
                wer = sum(wers)/len(wers)
                #print(f"cur wer: {wer}")

        
            wer = sum(wers)/len(wers)
            print(f"{subset} set wer: {wer}")

