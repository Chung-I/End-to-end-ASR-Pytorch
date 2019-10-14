from typing import Tuple, Dict, Any
from collections import OrderedDict
from functools import reduce

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from src.util import get_mask_from_sequence_lengths

Nonlinearities = {  # type: ignore
        "linear": lambda: lambda x: x,
        "relu": torch.nn.ReLU,
        "relu6": torch.nn.ReLU6,
        "elu": torch.nn.ELU,
        "prelu": torch.nn.PReLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "threshold": torch.nn.Threshold,
        "hardtanh": torch.nn.Hardtanh,
        "sigmoid": torch.nn.Sigmoid,
        "tanh": torch.nn.Tanh,
        "log_sigmoid": torch.nn.LogSigmoid,
        "softplus": torch.nn.Softplus,
        "softshrink": torch.nn.Softshrink,
        "softsign": torch.nn.Softsign,
        "tanhshrink": torch.nn.Tanhshrink,
}

def batch_norm_2d_factory(freq_dim, channel_dim, momentum=0.1, eps=1e-05):
    return nn.BatchNorm2d(channel_dim, momentum=momentum, eps=eps)

def seq_batch_norm_factory(freq_dim, channel_dim, momentum=0.1, eps=1e-05):
    return SeqBatchNorm(freq_dim * channel_dim, momentum=momentum, eps=eps)

BATCH_NORM = {
    "2d": batch_norm_2d_factory,
    "1d": seq_batch_norm_factory
}


def get_ds_ratio_factory(module):
    padding = module.padding[0] \
        if isinstance(module.padding, tuple) \
        else module.padding

    dilation = module.dilation[0] \
        if isinstance(module.dilation, tuple) \
        else module.dilation

    kernel_size = module.kernel_size[0] \
        if isinstance(module.kernel_size, tuple) \
        else module.kernel_size

    stride = module.stride[0] \
        if isinstance(module.stride, tuple) \
        else module.stride
    def get_ds_ratio(lengths):
        lengths = torch.floor(
            torch.clamp(
                (lengths.float() + 2 * padding - dilation *
                    (kernel_size - 1) - 1) / stride + 1,
                1.
            )
        ).long()
        return lengths

    return get_ds_ratio


class LengthAwareWrapper(nn.Module):
    def __init__(self, module, pass_through: str = False):
        super(LengthAwareWrapper, self).__init__()
        self.module = module
        self._pass_through = pass_through
        if not pass_through:
            self.padding = self.module.padding[0] \
                if isinstance(self.module.padding, tuple) \
                else self.module.padding

            self.dilation = self.module.dilation[0] \
                if isinstance(self.module.dilation, tuple) \
                else self.module.dilation

            self.kernel_size = self.module.kernel_size[0] \
                if isinstance(self.module.kernel_size, tuple) \
                else self.module.kernel_size

            self.stride = self.module.stride[0] \
                if isinstance(self.module.stride, tuple) \
                else self.module.stride

    def forward(self, inputs_and_lengths: Tuple[torch.FloatTensor, torch.LongTensor]):
        # pylint: disable=arguments-differ
        """
        Expect inputs of (N, C, T, D) dimension.
        There's something peculiar here in that we made the padding affects our length
        only at the start position, so that 2 * self.padding becomes 1 * self.padding.
        """
        inputs, lengths = inputs_and_lengths
        if not self._pass_through:
            lengths = torch.floor(
                torch.clamp(
                    (lengths.float() + 2 * self.padding - self.dilation *
                     (self.kernel_size - 1) - 1) / self.stride + 1,
                    1.
                )
            ).long()
        if getattr(self.module, 'forward', None):
            outputs = self.module.forward(inputs)
        else:
            outputs = self.module(inputs)
        return (outputs, lengths)


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(Conv1d, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1), "kernel_size should be odd if no given padding."
            padding = (dilation * (kernel_size - 1)) // 2

        self.conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.conv(x)


# https://github.com/mozilla/TTS/issues/26
class Prenet(nn.Module):
    def __init__(self, in_dim, hidden_dim=[256, 256], apply_dropout=0.5, norm_type=None):
        super(Prenet, self).__init__()
        input_dim = [in_dim] + hidden_dim[:-1]
        self.layers = nn.ModuleList([
            Linear(Din, Dout, bias=False, norm_type=norm_type) 
            for Din, Dout in zip(input_dim, hidden_dim)])
        self.relu = nn.ReLU()
        self.apply_dropout = apply_dropout
        
    def forward(self, x):
        """
        Arg:
            x: part of melspectrogram (batch, ..., n_frames_per_step * n_mels)
        Return:
            A tensor of shape (batch, ..., prenet_dim)
        """
        for layer in self.layers:
            # The dropout does NOT turn off even when doing evaluation.
            x = F.dropout(self.relu(layer(x)), p=self.apply_dropout, training=True)
        return x


class Postnet(nn.Module):
    """Postnet for tacotron2"""
    def __init__(self, n_mels, postnet_embed_dim, postnet_kernel_size, postnet_n_conv, postnet_dropout):
        super(Postnet, self).__init__()

        in_size = [n_mels] + [postnet_embed_dim] * (postnet_n_conv-1) 
        out_size = [postnet_embed_dim] * (postnet_n_conv-1) + [n_mels]
        act_fn = ['tanh'] * (postnet_n_conv-1) + ['linear']
        self.convs = nn.ModuleList([
            nn.Sequential(
                Conv1d(
                    in_channels=Din, 
                    out_channels=Dout,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=(postnet_kernel_size - 1) // 2,
                    dilation=1,
                    w_init_gain=fn),
                nn.BatchNorm1d(Dout),
                nn.Tanh() if fn == 'tanh' else nn.Identity(),
                nn.Dropout(postnet_dropout)) 
            for Din, Dout, fn in zip(in_size, out_size, act_fn)
        ])

    def forward(self, x):
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2)
        return x 


class Encoder(nn.Module):
    """Tacotron2 text encoder consists of convolution layers and bidirectional LSTM
    """
    def __init__(self, in_dim, enc_embed_dim, enc_n_conv, enc_rnn_layer, enc_kernel_size, enc_dropout=0.5):
        super(Encoder, self).__init__()
        in_size = [in_dim] + [enc_embed_dim] * (enc_n_conv-1) 
        out_size = [enc_embed_dim] * enc_n_conv
        self.convs = nn.ModuleList([
            nn.Sequential(
                Conv1d(in_channels=Din, 
                   out_channels=Dout,
                   kernel_size=enc_kernel_size,
                   stride=1,
                   padding=(enc_kernel_size - 1) // 2,
                   dilation=1,
                   w_init_gain='relu'),
                nn.BatchNorm1d(enc_embed_dim), 
                nn.ReLU(),
                nn.Dropout(enc_dropout)) 
            for Din, Dout in zip(in_size, out_size)
        ])
        self.lstm = nn.LSTM(
            input_size=enc_embed_dim, 
            hidden_size=enc_embed_dim // 2,
            num_layers=enc_rnn_layer,
            batch_first=True,
            bidirectional=True
        )        
        
    def forward(self, txt_embed, input_lengths):
        """
        Arg:
            txt_embed: torch.FloatTensor of shape (batch, L, enc_embed_dim)
        Return:
            The hidden representation of text, (batch, L, enc_embed_dim)
        """
        # (B, D, L)
        x = txt_embed.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        # (B, L, D)
        x = x.transpose(1, 2)
        
        #input_lengths = input_lengths.cpu().numpy()
        #x = nn.utils.rnn.pack_padded_sequence(
        #    x, input_lengths, batch_first=True, enforce_sorted=False)
        
        self.lstm.flatten_parameters()
        
        output, _ = self.lstm(x)
        #output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output


class Attention(nn.Module):
    """Atention module"""
    def __init__(self, query_dim, memory_dim, hidden_dim, 
                 n_location_filters, location_kernel_size, 
                 loc_aware, use_summed_weights):
        super(Attention, self).__init__()
        self.query_layer = Linear(query_dim, hidden_dim, 
                                  bias=False, w_init_gain='tanh')
        self.memory_layer = Linear(memory_dim, hidden_dim, 
                                   bias=False, w_init_gain='tanh')
        self.v = Linear(hidden_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        # loc: location-awared part
        self.loc_aware = loc_aware
        self.use_summed_weights = use_summed_weights
        if loc_aware:
            in_channels = 2 if use_summed_weights else 1
            self.loc_conv = Conv1d(in_channels=in_channels, 
                                   out_channels=n_location_filters, 
                                   kernel_size=location_kernel_size, 
                                   bias=False, stride=1, dilation=1)
            self.loc_linear = Linear(n_location_filters, hidden_dim, 
                                     bias=False, w_init_gain='tanh')

    def process_memory(self, memory):
        # Don't need to process memory for each decode timestep
        return self.memory_layer(memory)

    def energy(self, query, processed_memory, attn_history):
        """
        Arg:
            query: (B, out_dim of `prenet`)
            processed_memory: (B, L, hidden_dim)
            attn_history: (B, 2, L)
        Return:
            energy of shape (B, L)
        """
        processed_query = self.query_layer(query.unsqueeze(1))

        # Get location-awared feature
        if self.loc_aware:
            processed_loc_feat = self.loc_conv(attn_history).transpose(1, 2)
            processed_loc_feat = self.loc_linear(processed_loc_feat)
        else:
            processed_loc_feat = 0
        # Calculate energy
        energy = self.v(self.tanh(
            processed_query + processed_loc_feat + processed_memory))
        energy = energy.squeeze(-1)
        return energy

    def forward(self, query, memory, processed_memory, 
                attn_history, mask):
        energy = self.energy(query, processed_memory, attn_history)

        # Fill the -inf for padding encoder output
        if mask is not None:
            energy.data.masked_fill_(mask, -float('inf'))

        # (B, L)
        attn_weights = F.softmax(energy, dim=1)
        # Broadcast bmm, (B, 1, hidden_dim)
        attn_context = torch.bmm(attn_weights.unsqueeze(1), memory)
        attn_context = attn_context.squeeze(1)
        return attn_context, attn_weights


class Decoder(nn.Module):
    """Tacotron2 decoder consists of prenet, attention and LSTM"""
    def __init__(self, n_mels, n_frames_per_step, enc_embed_dim, prenet_dim, prenet_dropout,
                 query_rnn_dim, dec_rnn_dim, query_dropout, dec_dropout, 
                 attn_dim, n_location_filters, location_kernel_size, loc_aware, 
                 use_summed_weights, drop_dec_in, prenet_norm_type=None, pretrain=False):
        super(Decoder, self).__init__()
        self.n_mels = n_mels
        self.n_frames_per_step = n_frames_per_step
        self.enc_embed_dim = enc_embed_dim
        self.query_rnn_dim = query_rnn_dim
        self.dec_rnn_dim = dec_rnn_dim
        self.prenet_dropout = prenet_dropout
        self.prenet_dim = prenet_dim
        self.query_dropout = nn.Dropout(query_dropout)
        self.dec_dropout = nn.Dropout(dec_dropout)
        self.attn_dim = attn_dim
        self.n_location_filters = n_location_filters
        self.location_kernel_size = location_kernel_size
        self.pretrain = pretrain
        self.loc_aware = loc_aware
        self.use_summed_weights = use_summed_weights
        self.drop_dec_in = drop_dec_in
        self.prenet_norm_type = prenet_norm_type
        
        self.prenet = Prenet(
            n_mels * n_frames_per_step, [prenet_dim, prenet_dim], 
            apply_dropout=prenet_dropout, norm_type=prenet_norm_type)
        self.query_rnn = nn.LSTMCell(
            prenet_dim + enc_embed_dim, query_rnn_dim)
        self.attn = Attention(
            query_rnn_dim, enc_embed_dim, attn_dim, 
            n_location_filters, location_kernel_size,
            loc_aware, use_summed_weights)
        self.dec_rnn = nn.LSTMCell(
            query_rnn_dim + enc_embed_dim, dec_rnn_dim)
        self.proj = Linear(
            dec_rnn_dim + enc_embed_dim, n_mels * n_frames_per_step)
        self.gate_layer = Linear(
            dec_rnn_dim + enc_embed_dim, 1, bias=True, w_init_gain='sigmoid')

    def forward(self, memory, memory_lengths, teacher, tf_rate=0.0):
        """
        Arg:
            decoder_inputs: melspectrogram of shape (B, T, n_mels), None if inference
            memory: encoder outputs (B, L, D). It could be None if pretraining
            memory_lengths: the lengths of memory without padding, it could be None if pretraining
            teacher: melspectrogram provided as teacher or int. serving as max_dec_step
            max_dec_steps: Used only for inference.

        Return:
            mel_outputs: (B, T, n_mels)
            alignments: (B, T // n_frames_per_step, L)
            
        """
        # Init.
        device = memory.device
        B = memory.size(0)
        go_frame = torch.zeros( B, self.n_frames_per_step, self.n_mels, device=device)
        self.init_decoder_states(memory)
        mask = None # if self.pretrain else self._make_mask(memory_lengths)

        # Stage check
        inference = tf_rate==0.0
        if inference:
            decode_steps = teacher//self.n_frames_per_step
        else:
            # (B, T, n_mels) -> (B, T' (decode steps), n_frames_per_step, n_mels)
            decode_steps = teacher.shape[1]//self.n_frames_per_step
            teacher = teacher.view(B, decode_steps, -1)
            teacher = self.prenet(teacher)

        # Forward
        mel_outputs, alignments, stops = [], [], []
        dec_in = self.prenet(go_frame.view(B, -1))
        for t in range(decode_steps):
            mel_out, align, stop = self.decode_one_step(dec_in, mask)
            mel_outputs.append(mel_out)
            alignments.append(align)
            stops.append(stop)

            if inference or (np.random.rand()>tf_rate):
                # Use previous output
                dec_in = self.prenet(mel_out.view(B, self.n_frames_per_step * self.n_mels))
            elif np.random.rand()<self.drop_dec_in:
                dec_in = teacher.mean(dim=1)
            else:
                # Use ground truth
                dec_in = teacher[:, t, :]

        # (B, T, n_mels)
        mel_outputs = torch.cat(mel_outputs, dim=1)
        # (B, T', L)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (B, T)
        stops = torch.cat(stops, dim=1)
        return mel_outputs, alignments, stops

    def decode_one_step(self, dec_in, mask):
        """
        Arg:
            dec_in: melspectrogram of shape (B, n_frames_per_step * prenet_dim)
            mask: None if pretraining
        Return:
            mel_out: (B, n_frames_per_step, n_mels)
            attn_weights: (B, L)
        """
        B = dec_in.size(0)
        # For query_rnn
        query_rnn_input = torch.cat([dec_in, self.attn_context], dim=-1)
        hidden, cell = self.query_rnn(
            query_rnn_input, (self.query_rnn_hidden, self.query_rnn_cell))
        self.query_rnn_hidden = self.query_dropout(hidden)
        self.query_rnn_cell   = cell

        # Attention weights (for location-awared attention)
        # (B, 2, L)
        if self.use_summed_weights:
            attn_history = torch.stack([
                self.attn_weights, self.attn_weights_sum]).transpose(0, 1)
        else:
            attn_history = self.attn_weights.unsqueeze(1)
        # Perform attention
        if self.pretrain:
            ctx = torch.zeros_like(self.attn_context)
            weights = torch.zeros_like(self.attn_weights)
        else:
            ctx, weights = self.attn(
                query=self.query_rnn_hidden, 
                memory=self.memory, 
                processed_memory=self.processed_memory, 
                attn_history=attn_history, 
                mask=mask)
        self.attn_context = ctx
        self.attn_weights = weights
        self.attn_weights_sum = weights + self.attn_weights_sum
        # For dec_rnn
        dec_rnn_input = torch.cat([
            self.attn_context, self.query_rnn_hidden], dim=-1)
        hidden, cell = self.dec_rnn(
            dec_rnn_input, (self.dec_rnn_hidden, self.dec_rnn_cell))
        self.dec_rnn_hidden = self.dec_dropout(hidden)
        self.dec_rnn_cell   = cell
        # To predict mel output
        dec_rnn_hidden_attn_context = torch.cat([
            self.dec_rnn_hidden, 
            self.attn_context], dim=-1)
        mel_out = self.proj(dec_rnn_hidden_attn_context)
        mel_out = mel_out.view(B, self.n_frames_per_step, self.n_mels)
        stop = self.gate_layer(dec_rnn_hidden_attn_context).repeat(1, self.n_frames_per_step).sigmoid()
        return mel_out, self.attn_weights, stop
    
    def init_decoder_states(self, memory):
        B = memory.size(0)
        L = memory.size(1)
        device = memory.device
        # RNN states
        self.query_rnn_hidden = torch.zeros((B, self.query_rnn_dim), requires_grad=True, device=device)
        self.query_rnn_cell   = torch.zeros((B, self.query_rnn_dim), requires_grad=True, device=device)
        self.dec_rnn_hidden = torch.zeros((B, self.dec_rnn_dim), requires_grad=True, device=device)
        self.dec_rnn_cell   = torch.zeros((B, self.dec_rnn_dim), requires_grad=True, device=device)
        # Attention weights
        self.attn_weights     = torch.zeros((B, L), requires_grad=True, device=device)
        self.attn_weights_sum = torch.zeros((B, L), requires_grad=True, device=device)
        # Attention context
        self.attn_context = torch.zeros((B, self.enc_embed_dim), requires_grad=True, device=device)
        # Encoder output
        self.memory = memory
        self.processed_memory = self.attn.memory_layer(memory)

    @staticmethod
    def _make_mask(lengths):
        """
        Return:
            mask with 1 for not padded part and 0 for padded part
        """
        max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len).to(lengths.device)
        mask = (ids < lengths.unsqueeze(1)).bool()
        return ~mask


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear', norm_type=None):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(
            self.linear.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

        self.apply_norm = norm_type is not None
        if self.apply_norm:
            assert norm_type in ['LayerNorm','BatchNorm1d']
            self.norm_type = norm_type
            self.norm = getattr(nn,norm_type)(out_dim)#nn.BatchNorm1d(out_dim)
    
    def forward(self, x):
        x = self.linear(x)
        if self.apply_norm:
            if len(x.shape)==3 and self.norm_type=='BatchNorm1d':
                x = x.transpose(1,2)
            x = self.norm(x)
            if len(x.shape)==3 and self.norm_type=='BatchNorm1d':
                x = x.transpose(1,2)
        return x


class BatchNormConv1d(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride, padding, activation=None):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_size, momentum=0.99, eps=1e-3)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        x = self.activation(x) if self.activation is not None else x
        x = self.bn(x)
        return x


class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        H = self.relu(self.H(x))
        T = self.sigmoid(self.T(x))
        y = H * T + x * (1.0 - T)
        return y


class CBHG(nn.Module):
    """CBHG in original paper.
    Components:
        - 1-d convolution banks
        - highway networks
        - gru (bidirectional)
    """
    def __init__(self, in_dim, K=16, hidden_sizes=[128, 128]):
        super(CBHG, self).__init__()
        self.in_dim = in_dim
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList(
                [BatchNormConv1d(in_dim, in_dim, kernel_size=k, stride=1,
                    padding=k//2, activation=self.relu)
                for k in range(1, K+1)])
        self.pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        in_sizes = [K * in_dim] + hidden_sizes[:-1]
        activations = [self.relu] * (len(hidden_sizes) - 1) + [None]
        self.conv1d_projs = nn.ModuleList(
                [BatchNormConv1d(in_size, out_size, kernel_size=3,
                    stride=1, padding=1, activation=act)
                    for in_size, out_size, act in zip(in_sizes, hidden_sizes, activations)])

        self.pre_highway_proj = nn.Linear(hidden_sizes[-1], in_dim, bias=False)
        self.highways = nn.ModuleList(
                [Highway(in_dim, in_dim) for _ in range(4)])
        self.gru = nn.GRU(
                in_dim, in_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        x = inputs
        # Assert x's shape: (batch_size, timesteps, in_dim)
        assert x.size(-1) == self.in_dim
        # -> (batch_size, in_dim, timesteps)
        x = x.transpose(1, 2)
        T = x.size(-1)

        # -> (batch_size, in_dim * K, timesteps)
        x = torch.cat(
                [conv1d(x)[:, :, :T] for conv1d in self.conv1d_banks], dim=1)
        assert x.size(1) == self.in_dim * len(self.conv1d_banks)
        x = self.pool1d(x)[:, :, :T]

        for conv1d in self.conv1d_projs:
            x = conv1d(x)
        # -> (batch_size, timesteps, hidden_dim)
        x = x.transpose(1, 2)
        # -> (batch_size, timesteps, in_dim)
        x = self.pre_highway_proj(x)

        x = x+inputs
        for highway in self.highways:
            x = highway(x)

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False)
        # -> (batch_size, timesteps, 2 * in_dim)
        y, _ = self.gru(x)

        if input_lengths is not None:
            y, _ = nn.utils.rnn.pad_packed_sequence(
                    y, batch_first=True)
        return y



class PrenetTaco(nn.Module):
    def __init__(self, in_dim, hidden):
        super(PrenetTaco, self).__init__()
        self.layers = len(hidden)
        hid_dims = [in_dim] + hidden
        self.out_dim = hid_dims[-1]

        
        networks = nn.ModuleList()
        for l in range(self.layers):
            networks.append(
                nn.Sequential(
                    nn.Linear(hid_dims[l], hid_dims[l+1]),
                    nn.ReLU()))
        self.networks = networks

    def forward(self, x, dropout=True):
        for m in self.networks:
            x = m(x)
            x = F.dropout(x, p=0.5, training=dropout)
        return x
### ------------------------------------------- 
### -------------- For CBHG -------------------
### ------------------------------------------- 
class EncoderTaco(nn.Module):
    def __init__(self, in_dim, paras):
        super(EncoderTaco, self).__init__()

        self.in_dim = in_dim
        self.prenet = PrenetTaco(in_dim,**paras['prenet'])
        self.cbhg = CBHG(self.prenet.out_dim,**paras['cbhg'])

    def forward(self,inputs,input_len):
        outputs = self.prenet(inputs)
        return self.cbhg(outputs,input_len)


class DecoderTaco(nn.Module):
    def __init__(self, in_dim, n_frames_per_step, mem_stop_gradient):
        super(DecoderTaco, self).__init__()

        self.in_dim = in_dim
        self.r = n_frames_per_step
        self.mem_stop_gradient = mem_stop_gradient

        self.prenet = PrenetTaco(in_dim * self.r, [256, 128]) 

        attention_mech = BahdanauAttention
        self.attention_rnn = AttentionWrapper(
            nn.GRUCell(256 + 128, 256),
            attention_mech(256)
        )

        self.memory_layer = nn.Linear(256, 256, bias=False)
        self.project_to_decoder_in = nn.Linear(512, 256)

        self.decoder_rnns = nn.ModuleList( [nn.GRUCell(256, 256) for _ in range(2)])

        self.proj_to_mel = nn.Linear(256, in_dim * self.r)
        self.max_decoder_steps = 200

    def forward(self, encoder_outputs, teacher, tf_rate=0.0):
        B = encoder_outputs.size(0)
        device = encoder_outputs.device
        go_frame = torch.zeros( B, self.r*self.in_dim, device=device)
        if self.mem_stop_gradient:
            processed_memory = self.memory_layer(encoder_outputs.detach())
        else:
            processed_memory = self.memory_layer(encoder_outputs)

        # Run greedy decoding if inputs is None
        inference = tf_rate == 0.0
        if inference:
            decode_steps = teacher//self.r
        else:
            decode_steps = teacher.shape[1]//self.r
            teacher = teacher.view(B, decode_steps, -1)
            teacher = self.prenet(teacher, True) # ToDo : Always true?

        
        # Init decoder states
        attention_rnn_hidden = torch.zeros((B, 256)).to(encoder_outputs.device)
        decoder_rnn_hiddens = [torch.zeros((B, 256)).to(encoder_outputs.device) for _ in range(len(self.decoder_rnns))]
        current_attention = torch.zeros((B, 256)).to(encoder_outputs.device)


        mel_outputs, alignments = [], []
        dec_in = self.prenet(go_frame, True) # ToDo : Always true?
        for t in range(decode_steps):
            # Attention RNN
            attention_rnn_hidden, current_attention, align = self.attention_rnn(
                dec_in, current_attention, attention_rnn_hidden,
                encoder_outputs, processed_memory=processed_memory)

            # Concat RNN output and attention context vector
            dec_rnn_input = self.project_to_decoder_in(
                torch.cat((attention_rnn_hidden, current_attention), -1))

            # Pass through the decoder RNNs
            for idx in range(len(self.decoder_rnns)):
                decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](
                    dec_rnn_input, decoder_rnn_hiddens[idx])
                # Residual connectinon
                dec_rnn_input = decoder_rnn_hiddens[idx] + dec_rnn_input

            mel_out = self.proj_to_mel(dec_rnn_input)
            mel_outputs.append(mel_out)
            alignments.append(align)
            if inference and t > self.max_decoder_steps:
                break
            if inference or (np.random.rand()>tf_rate):
                # Use previous output
                dec_in = self.prenet(
                    mel_out.view(B, self.r * self.in_dim), True) # ToDo : Always true?
            else:
                # Use ground truth
                dec_in = teacher[:, t, :]

        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        alignments = torch.stack(alignments).transpose(0, 1)
        return mel_outputs, alignments

    def is_end_of_frames(self, output, eps=0.2):
        return (output.data <= eps).all()


class SeqBatchNorm(nn.Module):
    def __init__(self, input_size, **kwargs):
        super(SeqBatchNorm, self).__init__()
        self.input_size = input_size
        self.bn = nn.BatchNorm1d(self.input_size, **kwargs)

    def forward(self, inputs):
        batch_size, n_channels, timesteps, feat_dim = inputs.size()
        inputs = inputs.permute(0, 1, 3, 2).reshape(batch_size, -1, timesteps)
        inputs = self.bn(inputs)
        inputs = inputs.reshape(-1, n_channels, feat_dim, timesteps).permute(0, 1, 3, 2)
        return inputs


class VGGExtractor(nn.Module):
    ''' VGG extractor for ASR described in https://arxiv.org/pdf/1706.02737.pdf'''
    def __init__(self, freq_dim, in_channel, nonlinearity: str = 'relu',
                 batch_norm: Dict[str, Any] = None):
        super(VGGExtractor, self).__init__()
        self.init_dim = 64
        self.hide_dim = 128
        out_dim = (freq_dim // 4) * self.hide_dim
        self.in_channel = in_channel
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        non_linear = Nonlinearities[nonlinearity]()
        bn_type = batch_norm.pop('type') if batch_norm else None

        layers = [
            nn.Conv2d(in_channel, self.init_dim, 3, stride=1, padding=1),
            BATCH_NORM[bn_type](freq_dim, self.init_dim, **batch_norm) if batch_norm else None,
            non_linear,
            nn.Conv2d(self.init_dim, self.init_dim, 3, stride=1, padding=1),
            BATCH_NORM[bn_type](freq_dim, self.init_dim, **batch_norm) if batch_norm else None,
            non_linear,
            nn.MaxPool2d(2, stride=2), # Half-time dimension
            nn.Conv2d(self.init_dim, self.hide_dim, 3, stride=1, padding=1),
            BATCH_NORM[bn_type](freq_dim // 2, self.hide_dim, **batch_norm) if batch_norm else None,
            non_linear,
            nn.Conv2d(self.hide_dim, self.hide_dim, 3, stride=1, padding=1),
            BATCH_NORM[bn_type](freq_dim // 2, self.hide_dim, **batch_norm) if batch_norm else None,
            non_linear,
            nn.MaxPool2d(2, stride=2) # Half-time dimension
        ]

        layers = list(filter(lambda l: l is not None, layers))
        self.extractor = nn.Sequential(*layers)

    def check_dim(self,input_dim):
        # Check input dimension, delta feature should be stack over channel. 
        if input_dim%13 == 0:
            # MFCC feature
            return int(input_dim/13),13,(13//4)*self.hide_dim
        elif input_dim%40 == 0:
            # Fbank feature
            return int(input_dim/40),40,(40//4)*self.hide_dim
        else:
            raise ValueError('Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '+d)

    def view_input(self,feature,feat_len):
        # downsample time
        feat_len = feat_len//4
        # crop sequence s.t. t%4==0
        if feature.shape[1]%4 != 0:
            feature = feature[:,:-(feature.shape[1]%4),:].contiguous()
        bs,ts,ds = feature.shape
        # stack feature according to result of check_dim
        feature = feature.view(bs,ts,self.in_channel,self.freq_dim)
        feature = feature.transpose(1,2)

        return feature,feat_len

    def forward(self,feature,feat_len):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        #feature, feat_len = self.view_input(feature,feat_len)
        # Foward
        batch_size = feature.size(0)
        feature = feature.view(batch_size, -1, self.in_channel, self.freq_dim).permute(0, 2, 1, 3)
        for module in self.extractor._modules.values():        
            feature = module(feature)
            if isinstance(module, nn.MaxPool2d):
                feat_len = feat_len // 2

        batch_size, _, timesteps, _ = feature.size()
        feature = feature.permute(0, 2, 1, 3).reshape(batch_size, timesteps, -1)

        return feature, feat_len

    def get_hidden_states(self,feature,feat_len, layer_num=-1, layer_counter=0):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        #feature, feat_len = self.view_input(feature,feat_len)
        # Foward
        def flatten(feature):
            batch_size, _, timesteps, _ = feature.size()
            return feature.permute(0, 2, 1, 3).reshape(batch_size, timesteps, -1)

        batch_size = feature.size(0)
        feature = feature.view(batch_size, -1, self.in_channel, self.freq_dim).permute(0, 2, 1, 3)

        if layer_num == layer_counter - 1:
            return feature, feat_len, layer_counter - 1

        for idx, module in enumerate(self.extractor._modules.values()):
            feature = module(feature)
            if isinstance(module, nn.MaxPool2d):
                feat_len = feat_len // 2
            if layer_num == layer_counter + idx:
                flattened_features = flatten(feature)
                return flattened_features, feat_len, layer_counter + idx

        flattened_features = flatten(feature)
        return flattened_features, feat_len, layer_counter + idx


class FC_block(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None, dropout_rate=None):
        super(FC_block, self).__init__()
        self.m = nn.Linear(input_size, output_size, bias)
        self.act = activation
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

    def forward(self, x):
        h = self.m(x)
        if self.act:
            h = self.act(h)
        if self.dropout:
            h = self.dropout(h)
        return h


class ResBlock(nn.Module):
    def __init__(self, in_channels=[256, 256], out_channels=[256, 256], kernel_size=[5, 1], activation=nn.ReLU()):
        super(ResBlock, self).__init__()
        self.cn1 = Conv1dNorm(in_channels[0], out_channels[0], kernel_size[0], stride=1, padding=None)
        self.activation = activation
        self.cn2 = Conv1dNorm(in_channels[1], out_channels[1], kernel_size[1], stride=1, padding=None)
    def forward(self, x, lengths=None):
        y1, lengths = self.cn1(x, lengths)
        y1 = self.activation(y1)
        y2, lengths = self.cn2(y1, lengths)
        y2 = self.activation(y2)
        return y1 + y2, lengths


class Conv1dNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', bn=True):
        super(Conv1dNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))
        self.get_ds_ratio = get_ds_ratio_factory(self.conv)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, xlen=None):
        y = self.conv(x)
        if xlen is not None:
            xlen = self.get_ds_ratio(xlen)
            mask = get_mask_from_sequence_lengths(xlen, max(xlen))
            mask = mask.unsqueeze(1).expand_as(y)
            y = y.masked_fill(~mask, 0.0)
        if self.bn:
            y = self.bn(y)
        return y, xlen


class ResCNN(nn.Module):
    #in_dim : feature dim
    #out_dim: ctc output token dim
    def __init__(self, in_dim, num_layers, kernel_size, activation='ReLU', dropout_rate=0):
        super(ResCNN, self).__init__()
        self.feature_extractors =  nn.ModuleList([
            Conv1dNorm(in_dim, 256, kernel_size, stride=2, padding=None),
            Conv1dNorm(256, 256, kernel_size, stride=2, padding=None)
        ])

        self.blocks = nn.ModuleList([
            ResBlock(activation=getattr(nn, activation)())
            for _ in range(num_layers)])
        size_in = [256, 512, 512] ; size_out = [512, 512, 512]
        self.fc_layers = nn.ModuleList(
                [FC_block(si, so, activation=getattr(nn, activation)(), dropout_rate=dropout_rate) \
                        for si, so in zip(size_in, size_out)])

        self.out_dim = 512

    def forward(self, x, xlen=None):
        """
        Args:
            x: tensor with shape (batch_size, timesteps, input_size)
            input_length: the length of elements in x before padding
        Return:
            tensor with shape (batch_size, timesteps, output_size)
        """
        B, T, in_dim = x.shape
        y = x.permute(0, 2, 1)
        for extractor in self.feature_extractors:
            y, xlen = extractor(y, xlen)
        for block in self.blocks :
            y, xlen = block(y, xlen)
        y = y.permute(0 , 2, 1)
        for fc in self.fc_layers :
            y = fc(y)
        return y, xlen


class CNN(nn.Module):
    def __init__(self, freq_dim: int, in_channel: int, hidden_channel: int, num_layers: int,
                 kernel_size: int = 3, stride: int = 2, batch_norm: bool = False,
                 nonlinearity: str = 'linear'):
        super(CNN, self).__init__()
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_layers = num_layers
        self.freq_dim = freq_dim
        non_linear = Nonlinearities[nonlinearity]()
        layers = []
        for l in range(num_layers):
            in_channel = self.in_channel if l == 0 else self.hidden_channel
            conv = LengthAwareWrapper(nn.Conv2d(in_channel, self.hidden_channel,
                                                self.kernel_size, stride=self.stride,
                                                padding=1))
            layers.append(conv)
            if batch_norm:
                bn = LengthAwareWrapper(nn.BatchNorm2d(self.hidden_channel), pass_through=True)
                layers.append(bn)
            layers.append(LengthAwareWrapper(non_linear, pass_through=True))
        self.modules = nn.ModuleList(layers)
        strides = [self.module[idx].stride for idx in range(len(self.modules))
                   if hasattr(self.modules[idx], "stride")]
        self.downsample_rate = reduce(lambda x, y: x * y, strides)
        self.out_dim = self.get_output_dim()

    def get_input_dim(self) -> int:
        return self.in_channel

    def get_output_dim(self) -> int:
        return self.hidden_channel * self.freq_dim // self.downsample_rate

    def is_bidirectional(self) -> bool:
        return False

    def forward(self,feature,feat_len):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        #feature, feat_len = self.view_input(feature,feat_len)
        # Foward
        batch_size = feature.size(0)
        feature = feature.view(batch_size, -1, self.in_channel, self.freq_dim).permute(0, 2, 1, 3)
        for module in self.modules():        
            feature, feat_len = module((feature, feat_len))

        batch_size, _, timesteps, _ = feature.size()
        feature = feature.permute(0, 2, 1, 3).reshape(batch_size, timesteps, -1)

        return feature, feat_len

    def forward(self, feature, lengths):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        batch_size, timesteps, _ = feature.size()
        feature = feature.view(batch_size, -1, self.in_channel, self.freq_dim).permute(0, 2, 1, 3)
        hidden_outputs = []
        for module in self.modules:
            feature, lengths = module((feature, lengths))
            batch_size, _, timesteps, _ = feature.size()
            flattened_features = feature.permute(0, 2, 1, 3).reshape(batch_size, timesteps, -1)
            hidden_outputs.append((flattened_features, lengths))
        return hidden_outputs

    def get_hidden_states(self,feature,feat_len, layer_num=-1, layer_counter=0):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        #feature, feat_len = self.view_input(feature,feat_len)
        # Foward
        def flatten(feature):
            batch_size, _, timesteps, _ = feature.size()
            return feature.permute(0, 2, 1, 3).reshape(batch_size, timesteps, -1)

        batch_size = feature.size(0)
        feature = feature.view(batch_size, -1, self.in_channel, self.freq_dim).permute(0, 2, 1, 3)

        if layer_num == layer_counter - 1:
            return feature, feat_len, layer_counter - 1

        for idx, module in enumerate(self.modules):
            feature, feat_len = module(feature, feat_len)
            if layer_num == layer_counter + idx:
                flattened_features = flatten(feature)
                return flattened_features, feat_len, layer_counter + idx

        flattened_features = flatten(feature)
        return flattened_features, feat_len, layer_counter + idx

class RNNLayer(nn.Module):
    ''' RNN wrapper, includes time-downsampling'''
    def __init__(self, input_dim, module, dim, bidirection, dropout, layer_norm, sample_rate, sample_style, proj):
        super(RNNLayer, self).__init__()
        # Setup
        rnn_out_dim = 2*dim if bidirection else dim
        self.proj = proj
        stack_rate = sample_rate if sample_rate > 1 and sample_style == 'concat' else 1
        self.out_dim = stack_rate*rnn_out_dim if not self.proj else rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.sample_rate = sample_rate
        self.sample_style = sample_style

        if self.sample_style not in ['drop','concat']:
            raise ValueError('Unsupported Sample Style: '+self.sample_style)
        
        # Recurrent layer
        self.layer = getattr(nn,module.upper())(input_dim, dim, bidirectional=bidirection, num_layers=1, batch_first=True)

        # Regularizations
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout>0:
            self.dp = nn.Dropout(p=dropout)

        # Additional projection layer
        if self.proj:
            self.pj = nn.Linear(stack_rate*rnn_out_dim,rnn_out_dim)

    def forward(self, input_x , x_len):
        # Forward RNN
        if not self.training:
            self.layer.flatten_parameters()
        # ToDo: check time efficiency of pack/pad
        #input_x = pack_padded_sequence(input_x, x_len, batch_first=True, enforce_sorted=False)
        output,_ = self.layer(input_x)
        #output,x_len = pad_packed_sequence(output,batch_first=True)

        # Normalizations
        if self.layer_norm:
            output = self.ln(output)
        if self.dropout>0:
            output = self.dp(output)

        # Perform Downsampling
        if self.sample_rate > 1:
            batch_size,timestep,feature_dim = output.shape
            x_len = x_len//self.sample_rate

            if self.sample_style =='drop':
                # Drop the unselected timesteps
                output = output[:,::self.sample_rate,:].contiguous()
            else:
                # Drop the redundant frames and concat the rest according to sample rate
                if timestep%self.sample_rate != 0:
                    output = output[:,:-(timestep%self.sample_rate),:]
                output = output.contiguous().view(batch_size,int(timestep/self.sample_rate),feature_dim*self.sample_rate)

        if self.proj:
            output = torch.tanh(self.pj(output)) 

        return output,x_len


class BaseAttention(nn.Module):
    ''' Base module for attentions '''
    def __init__(self, temperature, num_head):
        super().__init__()
        self.temperature = temperature
        self.num_head = num_head
        self.softmax = nn.Softmax(dim=-1)
        self.reset_mem()

    def reset_mem(self):
        # Reset mask
        self.mask = None
        self.k_len = None

    def set_mem(self):
        pass

    def compute_mask(self,k,k_len):
        # Make the mask for padded states
        self.k_len = k_len
        bs,ts,_ = k.shape
        self.mask = np.zeros((bs,self.num_head,ts))
        for idx,sl in enumerate(k_len):
            self.mask[idx,:,sl:] = 1 # ToDo: more elegant way?
        self.mask = torch.from_numpy(self.mask).to(k_len.device, dtype=torch.bool).view(-1,ts)# BNxT

    def _attend(self, energy, value):
        attn = energy / self.temperature
        attn = attn.masked_fill(self.mask, -np.inf)
        attn = self.softmax(attn) # BNxT
        output = torch.bmm(attn.unsqueeze(1), value).squeeze(1) # BNxT x BNxTxD-> BNxD
        return output, attn


class ScaleDotAttention(BaseAttention):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, num_head):
        super().__init__(temperature, num_head)

    def forward(self, q, k, v):
        ts = k.shape[1]
        energy = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1) # BNxD * BNxDxT = BNxT
        output, attn = self._attend(energy,v)
        
        attn = attn.view(-1,self.num_head,ts) # BNxT -> BxNxT

        return output, attn


class LocationAwareAttention(BaseAttention):
    ''' Location-Awared Attention '''
    def __init__(self, kernel_size, kernel_num, dim, num_head, temperature):
        super().__init__(temperature, num_head)
        self.prev_att  = None
        self.loc_conv = nn.Conv1d(num_head, kernel_num, kernel_size=2*kernel_size+1, padding=kernel_size, bias=False)
        self.loc_proj = nn.Linear(kernel_num, dim,bias=False)
        self.gen_energy = nn.Linear(dim, 1)
        self.dim = dim

    def reset_mem(self):
        super().reset_mem()
        self.prev_att = None

    def set_mem(self, prev_att):
        self.prev_att = prev_att

    def forward(self, q, k, v):
        bs_nh,ts,_ = k.shape
        bs = bs_nh//self.num_head

        # Uniformly init prev_att
        if self.prev_att is None:
            self.prev_att = torch.zeros((bs,self.num_head,ts)).to(k.device)
            for idx,sl in enumerate(self.k_len):
                self.prev_att[idx,:,:sl] = 1.0/sl

        # Calculate location context
        loc_context = torch.tanh(self.loc_proj(self.loc_conv(self.prev_att).transpose(1,2))) # BxNxT->BxTxD
        loc_context = loc_context.unsqueeze(1).repeat(1,self.num_head,1,1).view(-1,ts,self.dim)   # BxNxTxD -> BNxTxD
        q = q.unsqueeze(1) # BNx1xD
        
        # Compute energy and context
        energy = self.gen_energy(torch.tanh( k+q+loc_context )).squeeze(2) # BNxTxD -> BNxT
        output, attn = self._attend(energy,v)
        attn = attn.view(bs,self.num_head,ts) # BNxT -> BxNxT
        self.prev_att = attn

        return output, attn
