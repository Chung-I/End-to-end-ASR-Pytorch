from functools import partial
import os
import math

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
CKPT_EPOCH = 20


class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        # Logger settings
        self.best_eer = 3.0
        self.best_tts_loss = float('inf')
        # Curriculum learning affects data loader
        self.curriculum = self.config['hparas']['curriculum']

    def fetch_data(self, data):
        ''' Move data to device and compute text seq. length'''
        _, feat, feat_len, txt, spkr_id = data
        if self.tts is not None and hasattr(self.tts, 'n_frames_per_step'):
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
        spkr_id = spkr_id.to(self.device)

        return feat, feat_len, txt, txt_len, spkr_id

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.tr_set, self.dv_set, self.tokenizer, self.audio_converter, msg, (self.spkr_weight, self.spkr_id_list) = load_dataset(
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
        self.asr, self.tts = None, None
        self.layer_num = self.config['tts']['layer_num']
        if self.layer_num > -2:
            self.asr = ASR(self.feat_dim, self.vocab_size, **
                           self.config['model']).to(self.device)
            with torch.no_grad():
                seq_len = 64
                n_channels = self.config['model']['delta'] + 1
                dummy_inputs = torch.randn(
                    (1, seq_len, n_channels * self.feat_dim)).to(self.device)
                dummy_feat_len = torch.full((1, ), seq_len)
                dummy_outs, dummy_out_len, _ = \
                    self.asr.encoder.get_hidden_states(
                        dummy_inputs, seq_len, self.layer_num)
                tts_upsample_rate = (
                    dummy_feat_len / dummy_out_len).int().item()
                tts_in_dim = dummy_outs.size(-1)
                id_in_dim = tts_in_dim

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
                self.tts = None

        id_in_dim = self.feat_dim if self.tts is not None else id_in_dim
        if self.config['id_net']['type'] == "netvlad":
            self.config['id_net'].pop('type')
            self.id_net = ThinResNet(
                self.spkr_num, **self.config['id_net']).to(self.device)
        else:
            self.config['id_net'].pop('type')
            self.id_net = RNNSimple(
                id_in_dim, self.spkr_num, **self.config['id_net']).to(self.device)

        # self.verbose(self.asr.create_msg())
        # self.verbose(self.tts.create_msg())
        if self.layer_num < -1:
            model_paras = [{'params': self.id_net.parameters()}]
        elif self.tts is None:
            model_paras = [{'params': self.asr.parameters()},
                           {'params': self.id_net.parameters()}]
            for param in self.asr.parameters():
                param.requires_grad = False
        else:
            model_paras = [{'params': self.asr.parameters()},
                           {'params': self.tts.parameters()},
                           {'params': self.id_net.parameters()}]
            for param in self.asr.parameters():
                param.requires_grad = False
            for param in self.tts.parameters():
                param.requires_grad = False

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

        if self.GRAD_CLIP < float('inf'):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.id_net.parameters(), self.GRAD_CLIP)
        else:
            grad_norm = get_grad_norm(self.id_net.parameters())

        if math.isnan(grad_norm):
            self.verbose('Error : grad norm is NaN @ step '+str(self.step))
        else:
            self.optimizer.step()
        self.timer.cnt('bw')
        return grad_norm

    def load_ckpt(self, cont=True):
        ''' Load ckpt if --load option is specified '''
        if self.paras.load:
            # Load weights
            ckpt = torch.load(
                self.paras.load, map_location=self.device if self.mode == 'train' else 'cpu')
            if self.asr is not None:
                self.asr.load_state_dict(ckpt['model'])
            if self.tts is not None:
                self.tts.load_state_dict(ckpt['tts'])
            # if self.amp:
            #    amp.load_state_dict(ckpt['amp'])
            # Load task-dependent items
            if self.mode == 'train':
                if cont:
                    self.id_net.load_state_dict(ckpt['id_net'])
                    self.step = ckpt['global_step']
                    self.optimizer.load_opt_state_dict(ckpt['optimizer'])
                    self.verbose('Load ckpt from {}, restarting at step {}'.format(
                        self.paras.load, self.step))
            else:
                raise NotImplementedError
                for k, v in ckpt.items():
                    if type(v) is float:
                        metric, score = k, v
                self.model.eval()
                if self.emb_decoder is not None:
                    self.emb_decoder.eval()
                self.verbose('Evaluation target = {} (recorded {} = {:.2f} %)'.format(
                    self.paras.load, metric, score))

    def save_checkpoint(self, f_name, metric, score):
        ''''
        Ckpt saver
            f_name - <str> the name phnof ckpt file (w/o prefix) to store, overwrite if existed
            score  - <float> The value of metric used to evaluate model
        '''
        ckpt_path = os.path.join(self.ckpdir, f_name)
        full_dict = {
            "id_net": self.id_net.state_dict(),
            "optimizer": self.optimizer.get_opt_state_dict(),
            "global_step": self.step,
            metric: score
        }
        # Additional modules to save
        # if self.amp:
        #    full_dict['amp'] = self.amp_lib.state_dict()
        if self.asr is not None:
            full_dict['asr'] = self.asr.state_dict()
        if self.tts is not None:
            full_dict['tts'] = self.tts.state_dict()

        torch.save(full_dict, ckpt_path)
        self.verbose("Saved checkpoint (step = {}, {} = {:.2f}) and status @ {}".
                     format(human_format(self.step), metric, score, ckpt_path))

    def exec(self):
        ''' Training identification neural network '''
        self.verbose('Total training steps {}.'.format(
            human_format(self.max_step)))
        #acc_loss, att_loss, emb_loss = None, None, None
        n_epochs = 0
        self.timer.set()
        loss_history, loss_history_1000, loss_history_epoch = [], [], []
        true_history, true_history_1000, true_history_epoch = [], [], []
        pred_history, pred_history_1000, pred_history_epoch = [], [], []

        while self.step < self.max_step:
            # Renew dataloader to enable random sampling
            if self.curriculum > 0 and n_epochs == self.curriculum:
                self.verbose(
                    'Curriculum learning ends after {} epochs, starting random sampling.'.format(n_epochs))
                self.tr_set, _, _, _, _, _, _, _ = \
                    load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory,
                                 False, **self.config['data'])
            for data in self.tr_set:
                # Pre-step : update tf_rate/lr_rate and do zero_grad
                tf_rate = self.optimizer.pre_step(self.step)
                total_loss = 0

                # Fetch data
                feat, feat_len, txt, txt_len, spkr_id = self.fetch_data(data)
                self.timer.cnt('rd')

                # Forward model
                # Note: txt should NOT start w/ <sos>
                with torch.no_grad():
                    if self.layer_num < -1:
                        hidden_outs = feat
                        hidden_len = feat_len
                    else:
                        deltas = self.asr.apply_delta_acceleration(feat)
                        hidden_outs, hidden_len, _ = self.asr.encoder.get_hidden_states(
                            deltas, feat_len, self.layer_num)
                        if self.tts is not None:
                            feat_pred, _, _ = self.tts(
                                hidden_outs, hidden_len, feat, tf_rate=tf_rate)
                            feat_pred_len = feat_pred.size(-2)
                            if not isinstance(self.tts, Tacotron2):
                                feat_pred = feat_pred.unsqueeze(1)
                            mask = get_mask_from_sequence_lengths(feat_len, feat_pred_len).unsqueeze(
                                1).unsqueeze(-1).expand_as(feat_pred).bool()
                            feat_pred = feat_pred.masked_fill(~mask, 0.0)
                            if self.step == 1:
                                tts_loss = self.freq_loss(
                                    feat_pred, feat[..., :feat_pred_len, :])

                id_in_feat = hidden_outs if self.tts is None else feat_pred
                spkr_embedding, spkr_pred = self.id_net(id_in_feat, hidden_len)
                if self.id_net.loss_fn == 'amsoftmax':
                    margin = 0.35
                    scale = 30
                    spkr_pred[np.arange(len(spkr_id)), spkr_id] -= margin
                    spkr_pred *= scale
                total_loss = F.cross_entropy(
                    spkr_pred, spkr_id, weight=self.spkr_weight)

                # Save training process of loss, spkr_pred, spkr_id
                loss = total_loss.cpu().item()
                loss_history.append(loss)
                # loss_history_1000.append(loss)
                loss_history_epoch.append(loss)
                pred = spkr_pred.cpu().argmax(dim=1).tolist()
                # pred_history.extend(pred)
                # pred_history_1000.extend(pred)
                pred_history_epoch.extend(pred)
                true = spkr_id.cpu().tolist()
                # true_history.extend(true)
                # true_history_1000.extend(true)
                true_history_epoch.extend(true)

                self.timer.cnt('fw')

                # Backprop
                grad_norm = self.backward(total_loss)
                self.step += 1

                #print(spkr_embedding[:3, -5:].data)
                # print(spkr_id[:3].data)
                #print(spkr_pred[:3, spkr_id[:3]].data)
                # print(spkr_pred[:3].max(dim=1))
                # print(grad_norm)

                # Logger
                if (self.step == 1) or (self.step % self.PROGRESS_STEP == 0):
                    loss = sum(loss_history)/len(loss_history)
                    self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.4f} | {}'
                                  .format(loss, grad_norm, self.timer.show()))
                    self.write_log('loss', {'tr_id': loss})
                    # self.write_log('cmatrix', [cm_figure(
                    #     true_history, pred_history, self.spkr_id_list)])
                    loss_history.clear()
                    # true_history.clear()
                    # pred_history.clear()
                    # if self.step % 1000 == 0:
                    #    loss_1000 = sum(loss_history_1000) / \
                    #        len(loss_history_1000)
                    #    self.write_log('loss', {'tr_id_1000': loss_1000})
                    #    self.write_log('cmatrix_1000', [cm_figure(
                    #        true_history_1000, pred_history_1000, self.spkr_id_list)])
                    #    loss_history_1000.clear()
                    #    true_history_1000.clear()
                    #    pred_history_1000.clear()

                    # self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.4f} | {}'
                    #              .format(total_loss.cpu().item(), grad_norm, self.timer.show()))
                    #self.write_log('loss', {'tr_id': total_loss})

                # Validation
                # if (self.step == 1) or (self.step % self.valid_step == 0):
                #    self.validate()

                # End of step
                # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                torch.cuda.empty_cache()
                self.timer.set()
                if self.step > self.max_step:
                    break
            n_epochs += 1
            loss_epoch = sum(loss_history_epoch) / len(loss_history_epoch)
            correctness = (np.array(true_history_epoch) ==
                           np.array(pred_history_epoch))
            acc_epoch = correctness.mean()
            self.write_log('loss', {'tr_id_epoch': loss_epoch})
            self.write_log('acc', {'acc_epoch': acc_epoch})
            self.write_log('cmatrix_epoch', [cm_figure(
                true_history_epoch, pred_history_epoch, self.spkr_id_list)])
            loss_history_epoch.clear()
            true_history_epoch.clear()
            pred_history_epoch.clear()

            if ((n_epochs > 1) and (n_epochs % CKPT_EPOCH == 0)):
                # Regular ckpt
                self.save_checkpoint('step_{}.pth'.format(
                    self.step), 'spkr_acc_epoch', acc_epoch)

        self.log.close()

    def validate(self):
        # Eval mode
        self.asr.eval()
        self.id_net.eval()
        if self.tts is not None:
            self.tts.eval()

        dev_spkr_emb, dev_spkr_id = [], []
        for i, data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i+1, len(self.dv_set)))
            # Fetch data
            feat, feat_len, txt, txt_len, spkr_id = self.fetch_data(data)
            dev_spkr_id += spkr_id.cpu().tolist()

            # Forward model
            with torch.no_grad():
                deltas = self.asr.apply_delta_acceleration(feat)
                hidden_outs, hidden_len, _ = self.asr.encoder.get_hidden_states(
                    deltas, feat_len, self.layer_num)
                if self.tts is not None:
                    feat_pred, align, _ = self.tts(
                        hidden_outs, hidden_len, feat.size(1), tf_rate=0.0)
                    feat_pred_len = feat_pred.size(-2)
                    if not isinstance(self.tts, Tacotron2):
                        feat_pred = feat_pred.unsqueeze(1)
                    mask = get_mask_from_sequence_lengths(feat_len, feat_pred_len)\
                        .unsqueeze(1).unsqueeze(-1).expand_as(feat_pred).bool()
                    feat_pred = feat_pred.masked_fill(~mask, 0.0)

                id_in_feat = hidden_outs if self.tts is None else feat_pred
                spkr_embedding, _ = self.id_net(id_in_feat)
                dev_spkr_emb += spkr_embedding.cpu().tolist()

                if self.step == 1 and i == len(self.dv_set)//2:
                    # TODO(Chung-I): unnecessary second forwarding, need to change
                    ctc_output, encode_len, att_output, att_align, dec_state = self.asr(
                        feat, feat_len, int(max(txt_len)*self.DEV_STEP_RATIO), emb_decoder=self.emb_decoder)

                    # Show some example on tensorboard
                    # pick n longest samples in the median batch
                    sample_txt = txt.cpu()[:DEV_N_EXAMPLES]
                    sample_mel = feat.cpu()[:DEV_N_EXAMPLES]
                    if ctc_output is not None:
                        ctc_hyps = ctc_output.argmax(
                            dim=-1).cpu()[:DEV_N_EXAMPLES]
                    else:
                        ctc_hyps = [None] * DEV_N_EXAMPLES
                    if att_output is not None:
                        att_hyps = att_output.argmax(
                            dim=-1).cpu()[:DEV_N_EXAMPLES]

                    for i, (mel, gt_txt, h_p) in enumerate(zip(sample_mel, sample_txt, ctc_hyps)):
                        if h_p is not None:
                            self.write_log('hyp_text{}'.format(i), self.tokenizer.decode(
                                h_p.tolist(), ignore_repeat=True))
                        self.write_log('truth_text{}'.format(
                            i), self.tokenizer.decode(gt_txt.tolist()))
                        self.write_log(
                            'mel_spec{}_gt'.format(i), feat_to_fig(mel))
                        self.write_log('mel_wave{}_gt'.format(
                            i), self.audio_converter.feat_to_wave(mel))

                    # Show tts example
                    if self.tts is not None:
                        tts_loss = self.freq_loss(
                            feat_pred, feat[..., :feat_pred_len, :])
                        channel = 1 if isinstance(
                            self.tts, Tacotron2) else 0
                        # PostNet product
                        mel_p = feat_pred.cpu()[:DEV_N_EXAMPLES, channel]
                        if align is None:
                            align_p = [None] * DEV_N_EXAMPLES
                        else:
                            align_p = align.cpu()[:DEV_N_EXAMPLES]

                        for i, (m_p, a_p) in enumerate(zip(mel_p, align_p)):
                            self.write_log(
                                'mel_spec{}'.format(i), feat_to_fig(m_p))
                            self.write_log('mel_wave{}'.format(
                                i), self.audio_converter.feat_to_wave(m_p))
                            if a_p is not None:
                                self.write_log(
                                    'dv_align{}'.format(i), feat_to_fig(a_p))

        assert len(dev_spkr_emb) == len(dev_spkr_id)
        eer, dcf2, dcf3 = roc_score(dev_spkr_emb, dev_spkr_id)

        if eer < self.best_eer:
            self.best_eer = eer
            if self.step > 1:
                self.save_checkpoint('id_net_{}.pth'.format(
                    self.step), 'spkr_eer', eer)

        if ((self.step > 1) and (self.step % CKPT_STEP == 0)):
            # Regular ckpt
            self.save_checkpoint('step_{}.pth'.format(
                self.step), 'spkr_eer', eer)

        self.write_log('ROC', {'EER': eer, 'DCF2': dcf2, 'DCF3': dcf3})

        # Resume training
        self.id_net.train()
