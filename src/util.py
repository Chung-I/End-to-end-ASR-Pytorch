from typing import Callable
from pathlib import Path
import os
import matplotlib.pyplot as plt
import math
import time
import itertools
from tqdm import tqdm
import torch.multiprocessing as mp

from numpy.lib.format import open_memmap
import torch
import numpy as np
from numpy.linalg import norm
from torch import nn
from torch._six import inf
import editdistance as ed
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')


class Timer():
    ''' Timer for recording training time distribution. '''

    def __init__(self):
        self.prev_t = time.time()
        self.clear()

    def set(self):
        self.prev_t = time.time()

    def cnt(self, mode):
        self.time_table[mode] += time.time()-self.prev_t
        self.set()
        if mode == 'bw':
            self.click += 1

    def show(self):
        total_time = sum(self.time_table.values())
        self.time_table['avg'] = total_time/self.click
        self.time_table['rd'] = 100*self.time_table['rd']/total_time
        self.time_table['fw'] = 100*self.time_table['fw']/total_time
        self.time_table['bw'] = 100*self.time_table['bw']/total_time
        msg = '{avg:.3f} sec/step (rd {rd:.1f}% | fw {fw:.1f}% | bw {bw:.1f}%)'.format(
            **self.time_table)
        self.clear()
        return msg

    def clear(self):
        self.time_table = {'rd': 0, 'fw': 0, 'bw': 0}
        self.click = 0

# Reference : https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/e2e_asr.py#L168


def init_weights(module):
    # Exceptions
    if type(module) == nn.Embedding:
        module.weight.data.normal_(0, 1)
    else:
        for p in module.parameters():
            data = p.data
            if data.dim() == 1:
                # bias
                data.zero_()
            elif data.dim() == 2:
                # linear weight
                n = data.size(1)
                stdv = 1. / math.sqrt(n)
                data.normal_(0, stdv)
            elif data.dim() in [3, 4]:
                # conv weight
                n = data.size(1)
                for k in data.size()[2:]:
                    n *= k
                stdv = 1. / math.sqrt(n)
                data.normal_(0, stdv)
            else:
                raise NotImplementedError


def init_gate(bias):
    n = bias.size(0)
    start, end = n // 4, n // 2
    bias.data[start:end].fill_(1.)
    return bias

# Convert Tensor to Figure on tensorboard


def feat_to_fig(feat):
    # feat TxD tensor
    feat = feat.transpose(1, 0)
    data = _save_canvas(feat.numpy())
    return torch.FloatTensor(data), "HWC"


def _save_canvas(data, meta=None):
    fig, ax = plt.subplots(figsize=(16, 8))
    if meta is None:
        ax.imshow(data, aspect="auto", origin="lower")
    else:
        ax.bar(meta[0], data[0], tick_label=meta[1], fc=(0, 0, 1, 0.5))
        ax.bar(meta[0], data[1], tick_label=meta[1], fc=(1, 0, 0, 0.5))
    fig.canvas.draw()
    # Note : torch tb add_image takes color as [0,1]
    data = np.array(fig.canvas.renderer._renderer)[:, :, :-1]/255.0
    plt.close(fig)
    return data

# Reference : https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings-in-python


def human_format(num):
    magnitude = 0
    while num >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '{:3.1f}{}'.format(num, [' ', 'K', 'M', 'G', 'T', 'P'][magnitude])


def cal_er(tokenizer, pred, truth, mode='wer', ctc=False):
    # Calculate error rate of a batch
    if pred is None:
        return np.nan
    elif len(pred.shape) >= 3:
        pred = pred.argmax(dim=-1)
    er = []
    for p, t in zip(pred, truth):
        p = tokenizer.decode(p.tolist(), ignore_repeat=ctc)
        t = tokenizer.decode(t.tolist())
        if mode == 'wer':
            p = p.split(' ')
            t = t.split(' ')
        er.append(float(ed.eval(p, t))/len(t))
    return sum(er)/len(er)


def load_embedding(text_encoder, embedding_filepath):
    with open(embedding_filepath, "r") as f:
        vocab_size, embedding_size = [int(x)
                                      for x in f.readline().strip().split()]
        embeddings = np.zeros((text_encoder.vocab_size, embedding_size))

        unk_count = 0

        for line in f:
            vocab, emb = line.strip().split(" ", 1)
            # fasttext's <eos> is </s>
            if vocab == "</s>":
                vocab = "<eos>"

            if text_encoder.token_type == "subword":
                idx = text_encoder.spm.piece_to_id(vocab)
            else:
                # get rid of <eos>
                idx = text_encoder.encode(vocab)[0]

            if idx == text_encoder.unk_idx:
                unk_count += 1
                embeddings[idx] += np.asarray([float(x)
                                               for x in emb.split(" ")])
            else:
                # Suppose there is only one (w, v) pair in embedding file
                embeddings[idx] = np.asarray(
                    [float(x) for x in emb.split(" ")])

        # Average <unk> vector
        if unk_count != 0:
            embeddings[text_encoder.unk_idx] /= unk_count

        return embeddings


def freq_loss(pred, label, sample_rate, n_mels, loss, differential_loss, emphasize_linear_low, p=1):
    """
    Args:
        pred: model output
        label: target
        loss: `l1` or `mse`
        differential_loss: use differential loss or not, see here `https://arxiv.org/abs/1909.10302`
        emphasize_linear_low: emphasize the low-freq. part of linear spectrogram or not

    Return:
        loss
    """
    # ToDo : Tao
    # pred -> BxTxD predicted mel-spec or linear-spec
    # label-> same shape
    # return loss for loss.backward()
    if loss == 'l1':
        criterion = torch.nn.functional.l1_loss
    elif loss == 'mse':
        criterion = torch.nn.functional.mse_loss
    else:
        raise NotImplementedError

    cutoff_freq = 3000

    # Repeat for postnet
    _, chn, _, dim = pred.shape
    label = label.unsqueeze(1).repeat(1, chn, 1, 1)

    loss_all = criterion(p * pred, p * label)

    if dim != n_mels and emphasize_linear_low:
        # Linear
        n_priority_freq = int(dim * (cutoff_freq / (sample_rate/2)))
        pred_low = pred[:, :, :, :n_priority_freq]
        label_low = label[:, :, :, :n_priority_freq]
        loss_low = criterion(p * pred_low, p * label_low)
        #loss_low = torch.nn.functional.mse_loss(p * pred_low, p * label_low)
        loss_all = 0.5 * loss_all + 0.5 * loss_low

    if differential_loss:
        pred_diff = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        label_diff = label[:, :, 1:, :] - label[:, :, :-1, :]
        loss_all += 0.5 * criterion(p * pred_diff, p * label_diff)

    return loss_all


def get_mask_from_sequence_lengths(sequence_lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Given a variable of shape ``(batch_size,)`` that represents the sequence lengths of each batch
    element, this function returns a ``(batch_size, max_length)`` mask variable.  For example, if
    our input was ``[2, 2, 3]``, with a ``max_length`` of 4, we'd return
    ``[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]``.

    We require ``max_length`` here instead of just computing it from the input ``sequence_lengths``
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return (sequence_lengths.unsqueeze(1) >= range_tensor).bool()


def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm


def mp_progress_map(func, arg_iter, num_workers):
    rs = []
    pool = mp.Pool(processes=num_workers)
    for args in arg_iter:
        rs.append(pool.apply_async(func, args))
    pool.close()

    rets = []
    for r in tqdm(rs):
        rets.append(r.get())
    pool.join()
    return rets


def wave_to_feat_and_save_factory(wave_to_feat: Callable):
    global func

    def func(wavfile):
        feat = wave_to_feat(wavfile)
        wavfile = Path(wavfile)
        feat_file = wavfile.with_suffix('.pt')
        aug_feat_file = None
        aug_feat_len = None
        if isinstance(feat, tuple):
            feat, aug_feat = feat
            aug_feat_len = aug_feat.shape[0]
            aug_feat_file = wavfile.with_suffix('.aug.pt')
            torch.save(aug_feat, aug_feat_file)
        feat_len = feat.shape[0]
        torch.save(feat, feat_file)
        return feat_len, feat_file, aug_feat_len, aug_feat_file

    return func


def write_sliced_array(sequence, out_path, total_len,
                       dtype='float32', func=None):
    print('writing into sliced array...')
    tbar = tqdm(total=len(sequence))
    it = ((func(x), x) if func else (x, x) for x in sequence)
    x, path = next(it)
    #shape = (x.shape[0], total_len)
    shape = (total_len, x.shape[1])
    data = open_memmap(out_path, mode='w+', dtype=dtype, shape=shape)

    def put(entry, prev):
        now = prev + entry.shape[0]
        #data[:, prev:now] = entry
        data[prev:now, :] = entry
        tbar.update(1)
        return now
    prev = put(x, 0)
    for x, path in it:
        prev = put(x, prev)
        if isinstance(path, Path):
            os.remove(path)
    tbar.close()
    print('flushing...')
    data.flush()
    print('done.')


def cm_figure(y_true, y_pred, class_names, normalize=False, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        # Normalize the confusion matrix.
        # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    figure = plt.figure(figsize=(64, 64))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        if cm[i, j] != 0:
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def roc_score(spkr_emb, spkr_id):
    score_matrix, label_matrix, true_match, false_match = [], [], [], []
    for i in range(len(spkr_id)):
        for j in range(i):
            #score = np.dot(embeddings[i, :], embeddings[j, :])
            a = spkr_emb[i][:]
            b = spkr_emb[j][:]
            score = np.dot(a, b)/(norm(a)*norm(b))
            score_matrix.append(score)
            if (spkr_id[i] == spkr_id[j]):
                label_matrix.append(1)
                true_match.append(score)
            else:
                label_matrix.append(0)
                false_match.append(score)
    #print("There are %d positive and %d negative pairs" %(len(true_match), len(false_match)))

    fpr, tpr, _ = roc_curve(label_matrix, score_matrix)
    intersection = abs(1-tpr-fpr)
    eer = fpr[np.argmin(intersection)]
    dcf2 = np.min(100*(0.01*(1-tpr)+0.99*fpr))
    dcf3 = np.min(1000*(0.001*(1-tpr)+0.999*fpr))
    return eer, dcf2, dcf3
