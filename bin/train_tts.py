from functools import partial
import os
import math

import torch
import torch.nn as nn
from src.solver import BaseSolver

from src.asr import ASR
from src.tts import FeedForwardTTS, HighwayTTS
from src.optim import Optimizer
from src.data import load_dataset
from src.util import human_format, cal_er, feat_to_fig, freq_loss, get_mask_from_sequence_lengths

DEV_N_EXAMPLES = 8 # How many examples to show in tensorboard
CKPT_STEP = 10000

class Solver(BaseSolver):
    ''' Solver for training'''
    def __init__(self,config,paras,mode):
        super().__init__(config,paras,mode)
        # Logger settings
        self.best_wer = {'att':3.0,'ctc':3.0}
        self.best_tts_loss = float('inf')
        # Curriculum learning affects data loader
        self.curriculum = self.config['hparas']['curriculum']

    def fetch_data(self, data):
        ''' Move data to device and compute text seq. length'''
        _, feat, feat_len, txt = data
        feat = feat.to(self.device)
        feat_len = feat_len.to(self.device)
        txt = txt.to(self.device)
        txt_len = torch.sum(txt!=0,dim=-1)
        
        return feat, feat_len, txt, txt_len

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.tr_set, self.dv_set, self.tokenizer, self.audio_converter, msg = \
                         load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory, 
                                      self.curriculum>0, **self.config['data'])
        self.vocab_size = self.tokenizer.vocab_size
        self.feat_dim, _ = self.audio_converter.feat_dim                  # ignore linear dim        
        self.verbose(msg)

    def set_model(self):
        ''' Setup ASR model and optimizer '''
        # Model
        self.model = ASR(self.feat_dim, self.vocab_size, **self.config['model']).to(self.device)
        # self.tts = TTS(self.feat_dim, None, self.config['tts'])
        self.layer_num = self.config['tts']['layer_num']

        if self.config['tts']['type'] == "linear":
            self.tts = FeedForwardTTS(self.model.encoder.layers[self.layer_num].out_dim,
                                      self.feat_dim, self.config['tts']['num_layers'],
                                      self.model.encoder.sample_rate).to(self.device)
        elif self.config['tts']['type'] == "highway":
            self.tts = HighwayTTS(self.model.encoder.layers[self.layer_num].out_dim,
                                  self.feat_dim, self.config['tts']['num_layers'],
                                  self.model.encoder.sample_rate).to(self.device)

        self.verbose(self.model.create_msg())
        #self.verbose(self.tts.create_msg())
        model_paras = [{'params':self.tts.parameters()}]

        # Losses
        self.freq_loss = partial(
            freq_loss, 
            sample_rate=self.audio_converter.sr, 
            n_mels=self.audio_converter.n_mels,
            loss=self.config['hparas']['freq_loss_type'],
            differential_loss=self.config['hparas']['differential_loss'],
            emphasize_linear_low=self.config['hparas']['emphasize_linear_low']
            )
        # Optimizer
        self.optimizer = Optimizer(model_paras, **self.config['hparas'])
        self.verbose(self.optimizer.create_msg())

        # Enable AMP if needed
        self.enable_apex()
        
        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt(cont=False)

    def backward(self, loss):
        '''
        Standard backward step with self.timer and debugger
        Arguments
            loss - the loss to perform loss.backward()
        '''
        self.timer.set()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.tts.parameters(), self.GRAD_CLIP)
        if math.isnan(grad_norm):
            self.verbose('Error : grad norm is NaN @ step '+str(self.step))
        else:
            self.optimizer.step()
        self.timer.cnt('bw')
        return grad_norm

    def save_checkpoint(self, f_name, metric, score):
        '''' 
        Ckpt saver
            f_name - <str> the name phnof ckpt file (w/o prefix) to store, overwrite if existed
            score  - <float> The value of metric used to evaluate model
        '''
        ckpt_path = os.path.join(self.ckpdir, f_name)
        full_dict = {
            "model": self.tts.state_dict(),
            "optimizer": self.optimizer.get_opt_state_dict(),
            "global_step": self.step,
            metric: score
        }
        # Additional modules to save
        #if self.amp:
        #    full_dict['amp'] = self.amp_lib.state_dict()
        if self.emb_decoder is not None:
            full_dict['emb_decoder'] = self.emb_decoder.state_dict()

        torch.save(full_dict, ckpt_path)
        self.verbose("Saved checkpoint (step = {}, {} = {:.2f}) and status @ {}".\
                                       format(human_format(self.step),metric,score,ckpt_path))

    def exec(self):
        ''' Training End-to-end ASR system '''
        self.verbose('Total training steps {}.'.format(human_format(self.max_step)))
        ctc_loss, att_loss, emb_loss = None, None, None
        n_epochs = 0
        self.timer.set()

        while self.step< self.max_step:
            # Renew dataloader to enable random sampling 
            if self.curriculum>0 and n_epochs==self.curriculum:
                self.verbose('Curriculum learning ends after {} epochs, starting random sampling.'.format(n_epochs))
                self.tr_set, _, _, _, _, _ = \
                         load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory, 
                                      False, **self.config['data'])
            for data in self.tr_set:
                # Pre-step : update tf_rate/lr_rate and do zero_grad
                tf_rate = self.optimizer.pre_step(self.step)
                total_loss = 0
                
                # Fetch data
                feat, feat_len, txt, txt_len = self.fetch_data(data)
                self.timer.cnt('rd')

                # Forward model
                # Note: txt should NOT start w/ <sos>
                with torch.no_grad():
                    ctc_output, encode_len, enc_hiddens, att_output, att_align, dec_state = \
                        self.model(feat, feat_len, max(txt_len), tf_rate=1.0,
                                   teacher=txt)

                feat_pred = self.tts(enc_hiddens[self.layer_num][0])
                mask = get_mask_from_sequence_lengths(feat_len, max(feat_len))[:, :feat_pred.size(1)]\
                    .unsqueeze(-1).expand_as(feat_pred).bool()
                feat_pred = feat_pred.masked_fill(mask, 0.0)
                tts_loss = self.freq_loss(feat_pred.unsqueeze(1), feat[:, :feat_pred.size(1)])
                total_loss = tts_loss

                self.timer.cnt('fw')

                # Backprop
                grad_norm = self.backward(total_loss)
                self.step+=1

                # Logger
                if (self.step==1) or (self.step%self.PROGRESS_STEP==0):
                    self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'\
                            .format(total_loss.cpu().item(),grad_norm,self.timer.show()))
                    self.write_log('loss',{'tr_tts':tts_loss})

                # Validation
                if (self.step==1) or (self.step%self.valid_step == 0):
                    self.validate()

                # End of step
                torch.cuda.empty_cache() # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                self.timer.set()
                if self.step > self.max_step:break
            n_epochs +=1
        self.log.close()
        
    def validate(self):
        # Eval mode
        self.model.eval()
        self.tts.eval()
        dev_tts_loss = []

        for i,data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i+1,len(self.dv_set)))
            # Fetch data
            feat, feat_len, txt, txt_len = self.fetch_data(data)

            # Forward model
            with torch.no_grad():
                ctc_output, encode_len, enc_hiddens, att_output, att_align, dec_state = \
                    self.model(feat, feat_len, max(txt_len),
                                teacher=txt)
                feat_pred = self.tts(enc_hiddens[self.layer_num][0])
                #mask = get_mask_from_sequence_lengths(feat_len, max(feat_len))
                tts_loss = self.freq_loss(feat_pred.unsqueeze(1), feat[:, :feat_pred.size(1)]) # * mask[:, :feat_pred.size(1)]
            dev_tts_loss.append(tts_loss)

            # Show some example on tensorboard
            if i == len(self.dv_set)//2:
                # pick n longest samples in the median batch
                sample_txt = txt.cpu()[:DEV_N_EXAMPLES]
                if ctc_output is not None:
                    ctc_hyps = ctc_output.argmax(dim=-1).cpu()[:DEV_N_EXAMPLES]
                if att_output is not None:
                    att_hyps = att_output.argmax(dim=-1).cpu()[:DEV_N_EXAMPLES]
                mel_p = feat_pred.cpu()[:DEV_N_EXAMPLES] # PostNet product
                #align_p = align.cpu()[:DEV_N_EXAMPLES]
                sample_mel = feat.cpu()[:DEV_N_EXAMPLES]

        for i,(m_p,h_p) in enumerate(zip(mel_p, ctc_hyps)):
            self.write_log('hyp_text{}'.format(i), self.tokenizer.decode(h_p.tolist(), ignore_repeat=True))
            self.write_log('mel_spec{}'.format(i), feat_to_fig(m_p))
            self.write_log('mel_wave{}'.format(i), self.audio_converter.feat_to_wave(m_p))
            # self.write_log('dv_align{}'.format(i), feat_to_fig(a_p))

        if self.step ==1:
            for i,(mel,gt_txt) in enumerate(zip(sample_mel, sample_txt)):
                self.write_log('truth_text{}'.format(i), self.tokenizer.decode(gt_txt.tolist()))
                self.write_log('mel_spec{}_gt'.format(i), feat_to_fig(mel))
                self.write_log('mel_wave{}_gt'.format(i), self.audio_converter.feat_to_wave(mel))

        # Ckpt if performance improves 
        dev_tts_loss = sum(dev_tts_loss)/len(dev_tts_loss)

        if dev_tts_loss < self.best_tts_loss:
            self.best_tts_loss = dev_tts_loss
            if self.step>1:
                self.save_checkpoint('tts_{}.pth'.format(self.step), 'tts_loss', dev_tts_loss)

        if ((self.step>1) and (self.step % CKPT_STEP == 0)):
            # Regular ckpt
            self.save_checkpoint('step_{}.pth'.format(self.step), 'tts_loss', dev_tts_loss)

        self.write_log('speech_loss',{'dev':dev_tts_loss})

        # Resume training
        self.tts.train()
