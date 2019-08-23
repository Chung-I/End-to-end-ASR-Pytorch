import time
from overrides import overrides
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from src.module import BaseAttention
import pdb


def cuda_benchmark(func, *args, **kwargs):
    torch.cuda.synchronize()
    start = time.time()
    results = func(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()
    print("execution time: {}".format(end - start))
    return results


def cumprod(x, dim=-1, exclusive=False):
    """Numerically stable cumulative product by cumulative sum in log-space"""
    if exclusive:
        length = x.size(dim)
        x = torch.narrow(F.pad(x, pad=(1, 0, 0, 0), value=1.0), dim, 0, length)
    return torch.cumprod(x, dim=dim)


def moving_sum(x, back, forward):
    """Parallel moving sum with 1D Convolution"""
    # Pad window before applying convolution
    # [batch_size,    back + sequence_length + forward]
    x_padded = F.pad(x, pad=(back, forward))

    # Fake channel dimension for conv1d
    # [batch_size, 1, back + sequence_length + forward]
    x_padded = x_padded.unsqueeze(1)

    # Apply conv1d with filter of all ones for moving sum
    filters = x.new_ones(1, 1, back + forward + 1)

    x_sum = F.conv1d(x_padded, filters)

    # Remove fake channel dimension
    # [batch_size, sequence_length]
    return x_sum.squeeze(1)


def moving_max(x, w):
    """Compute the moving sum of x over a window with the provided bounds.

    x is expected to be of shape (batch_size, sequence_length).
    The returned tensor x_max is computed as
    x_max[i, j] = max(x[i, j - window + 1], ..., x[i, j])
    """
    # Pad x with -inf at the start
    x = F.pad(x, pad=(w - 1, 0), value=float("-inf"))
    # Add "channel" dimension (max_pool operates on 1D)
    x = x.unsqueeze(1)
    x = F.max_pool1d(x, kernel_size=w, stride=1).squeeze(1)
    return x


def fliped_cumsum(tensor, dim=-1):
    return torch.flip(torch.cumsum(torch.flip(tensor, dims=[dim]), dim=dim), dims=[dim])


def frame(tensor, chunk_size, pad_end=False, value=0):
    if pad_end:
        padded_tensor = F.pad(tensor, pad=(0, chunk_size - 1), value=value)
    else:
        padded_tensor = F.pad(tensor, pad=(chunk_size - 1, 0), value=value)
    framed_tensor = F.unfold(padded_tensor.unsqueeze(1).unsqueeze(-1),
                             kernel_size=(chunk_size, 1)).transpose(-2, -1)
    return framed_tensor


def soft_efficient(p_select, previous_alpha):
    cumprod_1_minus_p = cumprod(1 - p_select, dim=-1, exclusive=True)

    alpha = p_select * cumprod_1_minus_p * \
        torch.cumsum(previous_alpha /
                     torch.clamp(cumprod_1_minus_p, 1e-20, 1.), dim=1)
    return alpha


class Energy(nn.Module):
    def __init__(self,
                 enc_dim: int,
                 dec_dim: int,
                 att_dim: int,
                 mode: str = "bahdanau",
                 init_r: float = -1) -> None:
        """
        [Modified Bahdahnau attention] from
        "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
        http://arxiv.org/abs/1704.00784

        Used for Monotonic Attention and Chunk Attention
        """
        super().__init__()
        self.tanh = nn.Tanh()
        self.W = nn.Linear(enc_dim, att_dim, bias=False)
        #self.V = nn.Linear(dec_dim, att_dim, bias=False)
        self.b = nn.Parameter(torch.Tensor(att_dim).normal_())

        self.v = nn.utils.weight_norm(nn.Linear(att_dim, 1))
        self.v.weight_g = nn.Parameter(torch.Tensor([1 / att_dim]).sqrt())

        self.r = nn.Parameter(torch.Tensor([init_r]))
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, decoder_h, key, encoder_outputs):
        """
        Args:
            encoder_outputs: [batch_size, sequence_length, enc_dim]
            decoder_h: [batch_size, dec_dim]
        Return:
            Energy [batch_size, sequence_length]
        """
        batch_size, sequence_length, enc_dim = encoder_outputs.size()
        q = self.W(decoder_h)
        encoder_outputs = encoder_outputs.reshape(-1, enc_dim)
        energy = self.tanh(torch.bmm(q.unsqueeze(1),
                                     key.transpose(1, 2)).squeeze(1) + self.b)  # BNxD * BNxDxT = BNxT
        energy = self.v(energy).squeeze(-1) + self.r

        energy = energy.view(batch_size, sequence_length)
        return energy.masked_fill((1 - self.mask).byte(), float('-inf'))


class MonotonicAttention(nn.Module):
    def __init__(self,
                 enc_dim: int,
                 dec_dim: int,
                 att_dim: int,
                 temperature: float = 1.0,
                 num_head: int = 1,
                 dirac_at_first_step: bool = True):
        """
        [Monotonic Attention] from
        "Online and Linear-Time Attention by Enforcing Monotonic Alignment" (ICML 2017)
        http://arxiv.org/abs/1704.00784
        """

        super().__init__()
        self._dirac_at_first_step = dirac_at_first_step
        self.monotonic_energy = Energy(enc_dim, dec_dim, att_dim)
        self.prev_att = None
        self.reset_mem()

    def reset_mem(self):
        # Reset mask
        self.mask = None
        self.k_len = None
        self.monotonic_energy.set_mask(self.mask)

    def compute_mask(self, k_len):
        # Make the mask for padded states
        self.k_len = k_len
        bs = len(k_len)
        ts = max(k_len)
        self.mask = np.zeros((bs, self.num_head, ts))
        for idx, sl in enumerate(k_len):
            self.mask[idx, :, sl:] = 1  # ToDo: more elegant way?
        self.mask = torch.from_numpy(self.mask).to(
            k_len.device, dtype=torch.bool).view(-1, ts)  # BNxT
        self.monotonic_energy.set_mask(self.mask)

    def gaussian_noise(self, tensor):
        """Additive gaussian nosie to encourage discreteness"""
        return tensor.new_empty(tensor.size()).normal_()

    def soft_recursive(self, decoder_h, key, encoder_outputs):
        """
        Soft monotonic attention (Train)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_alpha [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
        """
        batch_size, sequence_length, _ = encoder_outputs.size()
        end_mask = self.mask * \
            F.pad((1 - self.mask), pad=(0, 1), value=1.)[:, 1:]

        monotonic_energy = self.monotonic_energy(
            decoder_h, key, encoder_outputs)
        p_select = F.sigmoid(monotonic_energy +
                             self.gaussian_noise(monotonic_energy))
        p_select = torch.where(end_mask.byte(), end_mask, p_select)

        shifted_1mp_choose_i = F.pad(
            1 - p_select[:, :-1], pad=(1, 0, 0, 0), value=1.0)

        if self.prev_att is None:
            if self._dirac_at_first_step:
                alpha = decoder_h.new_zeros(batch_size, sequence_length)
                alpha[:, 0] = 1.0
            else:
                cumprod_1_minus_p = cumprod(
                    1 - p_select, dim=-1, exclusive=True)
                alpha = p_select * cumprod_1_minus_p

        else:
            alpha_div_ps = []
            alpha_div_p = self.prev_att.new_zeros(batch_size)
            for j in range(sequence_length):
                alpha_div_p = shifted_1mp_choose_i[:, j] * \
                    alpha_div_p + self.prev_att[:, j]
                alpha_div_ps.append(alpha_div_p)
            alpha = p_select * torch.stack(alpha_div_ps, -1)

        self.prev_att = alpha

        return alpha

    def soft(self, decoder_h, key, encoder_outputs):
        """
        Soft monotonic attention (Train)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_alpha [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
        """
        batch_size, sequence_length, _ = encoder_outputs.size()
        end_mask = self.mask * \
            F.pad((1 - self.mask), pad=(0, 1), value=1.)[:, 1:]

        monotonic_energy = self.monotonic_energy(
            decoder_h, key, encoder_outputs)
        p_select = F.sigmoid(monotonic_energy +
                             self.gaussian_noise(monotonic_energy))
        p_select = torch.where(end_mask.byte(), end_mask, p_select)

        # cumprod_1_minus_p = cumprod(1 - p_select, dim=-1, exclusive=True)
        if self.prev_att is None:
            if self._dirac_at_first_step:
                alpha = decoder_h.new_zeros(batch_size, sequence_length)
                alpha[:, 0] = 1.0
            else:
                cumprod_1_minus_p = cumprod(
                    1 - p_select, dim=-1, exclusive=True)
                alpha = p_select * cumprod_1_minus_p

        else:
            alpha = soft_efficient(p_select, self.prev_att)

        self.prev_att = alpha

        return alpha

    def hard(self, decoder_h, key, encoder_outputs):
        """
        Hard monotonic attention (Test)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_attention [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
        """

        batch_size, sequence_length, _ = encoder_outputs.size()

        if self.prev_att is None and self._dirac_at_first_step:
            # First iteration => alpha = [1, 0, 0 ... 0]
            self.prev_att = decoder_h.new_zeros(batch_size, sequence_length)
            self.prev_att = decoder_h.new_ones(batch_size)
        else:
            # TODO: Linear Time Decoding
            # It's not clear if authors' TF implementation decodes in linear time.
            # https://github.com/craffel/mad/blob/master/example_decoder.py#L235
            # They calculate energies for whole encoder outputs
            # instead of scanning from previous attended encoder output.
            monotonic_energy = self.monotonic_energy(
                decoder_h, key, encoder_outputs)

            # Hard Sigmoid
            # Attend when monotonic energy is above threshold (Sigmoid > 0.5)
            above_threshold = (monotonic_energy > 0).float()
            if self.prev_att is None:
                p_select = above_threshold
            else:
                p_select = above_threshold * \
                    torch.cumsum(self.prev_att, dim=1)
            attention = p_select * cumprod(1 - p_select, exclusive=True)

            # Not attended => attend at last encoder output
            # Assume that encoder outputs are not padded
            end_mask = self.mask * \
                F.pad((1 - self.mask), pad=(0, 1), value=1.)[:, 1:]

            attended = attention.sum(dim=1)
            attention.masked_fill_(
                (end_mask * (1 - attended.unsqueeze(-1))).byte(), 1.0)

            # Ex)
            # p_select                        = [0, 0, 0, 1, 1, 0, 1, 1]
            # 1 - p_select                    = [1, 1, 1, 0, 0, 1, 0, 0]
            # exclusive_cumprod(1 - p_select) = [1, 1, 1, 1, 0, 0, 0, 0]
            # attention: product of above     = [0, 0, 0, 1, 0, 0, 0, 0]
        self.prev_att = attention
        return attention

    @overrides
    def forward(self, q, k, v, mode="soft"):
        if mode not in ["soft", "recursive", "hard"]:
            raise ValueError("Invalid forward mode {} for attention; \
                accept only soft and hard mode".format(mode))
        att_func = {"soft": self.soft,
                    "recursive": self.recursive, "hard": self.hard}
        return att_func[mode](q, k, v)


class MoChA(nn.Module):
    def __init__(self,
                 chunk_size: int,
                 enc_dim: int,
                 dec_dim: int,
                 att_dim: int,
                 temperature: float = 1.0,
                 num_head: int = 1,
                 dirac_at_first_step: bool = False) -> None:
        """
        [Monotonic Chunkwise Attention] from
        "Monotonic Chunkwise Attention" (ICLR 2018)
        https://openreview.net/forum?id=Hko85plCW
        """
        super().__init__()
        self._monotonic_attention = MonotonicAttention(enc_dim, dec_dim, att_dim,
                                                       temperature, num_head, dirac_at_first_step)
        self.num_head = num_head
        self.chunk_size = chunk_size
        self.chunk_energy = Energy(enc_dim, dec_dim, att_dim)
        self.unfold = nn.Unfold(kernel_size=(self.chunk_size, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.reset_mem()

    def reset_mem(self):
        # Reset mask
        self.mask = None
        self.k_len = None
        self._monotonic_attention.reset_mem()

    def compute_mask(self, k_len):
        # Make the mask for padded states
        self.k_len = k_len
        bs = len(k_len)
        ts = max(k_len)
        self.mask = np.zeros((bs, self.num_head, ts))
        for idx, sl in enumerate(k_len):
            self.mask[idx, :, sl:] = 1  # ToDo: more elegant way?
        self.mask = torch.from_numpy(self.mask).to(
            k_len.device, dtype=torch.bool).view(-1, ts)  # BNxT
        self._monotonic_attention.compute_mask(k_len)

    def my_soft(self, emit_probs, chunk_energy):
        """
        PyTorch version of stable_chunkwise_attention in author's TF Implementation:
        https://github.com/craffel/mocha/blob/master/Demo.ipynb.
        Compute chunkwise attention distribution stably by subtracting logit max.
        """
        batch_size, _ = emit_probs.size()
        framed_chunk_energy = frame(
            chunk_energy, self.chunk_size, value=float("-inf"))

        chunk_probs = F.softmax(framed_chunk_energy, dim=-1)

        non_inf_mask = 1 - (framed_chunk_energy == float("-inf"))
        chunk_probs = torch.where(
            non_inf_mask, chunk_probs, non_inf_mask.float())

        weighted_chunk_probs = emit_probs.unsqueeze(-1) * chunk_probs
        kernel = torch.eye(
            self.chunk_size, dtype=chunk_probs.dtype, device=chunk_probs.device)
        kernel = torch.flip(kernel, dims=(-1,)).flatten()
        padded_chunk_probs = F.pad(
            weighted_chunk_probs, pad=(0, 0, 0, self.chunk_size - 1))
        beta = F.conv1d(padded_chunk_probs.view(batch_size, 1, -1),
                        kernel.view(1, 1, -1),
                        stride=self.chunk_size).squeeze(1)
        return beta

    def stable_soft(self, emit_probs, chunk_energy):
        """
        PyTorch version of stable_chunkwise_attention in author's TF Implementation:
        https://github.com/craffel/mocha/blob/master/Demo.ipynb.
        Compute chunkwise attention distribution stably by subtracting logit max.
        """
        # Compute length-chunk_size sliding max of sequences in softmax_logits (m)
        # (batch_size, sequence_length)
        chunk_energy_max = moving_max(chunk_energy, self.chunk_size)
        framed_chunk_energy = frame(
            chunk_energy, self.chunk_size, value=float("-inf"))
        framed_chunk_energy = framed_chunk_energy - \
            chunk_energy_max.unsqueeze(-1)
        softmax_denominators = torch.sum(torch.exp(framed_chunk_energy), -1)
        # Construct matrix of framed denominators, padding at the end so the final
        # frame is [softmax_denominators[-1], inf, inf, ..., inf] (E)
        framed_denominators = frame(
            softmax_denominators, self.chunk_size, pad_end=True, value=float("inf"))
        framed_chunk_energy_max = frame(chunk_energy_max, self.chunk_size, pad_end=True,
                                        value=float("inf"))
        softmax_numerators = torch.exp(
            chunk_energy.unsqueeze(-1) - framed_chunk_energy_max)

        framed_probs = frame(emit_probs, self.chunk_size, pad_end=True)
        beta = torch.sum(framed_probs * softmax_numerators /
                         framed_denominators, dim=-1)
        beta = torch.where(beta != beta, beta.new_zeros(beta.size()), beta)
        return beta

    def soft(self, alpha, u):
        """
        Args:
            alpha [batch_size, sequence_length]: emission probability in monotonic attention
            u [batch_size, sequence_length]: chunk energy
            chunk_size (int): window size of chunk
        Return
            beta [batch_size, sequence_length]: MoChA weights
        """

        # Numerical stability
        # Divide by same exponent => doesn't affect softmax
        u -= torch.max(u, dim=1, keepdim=True)[0]
        exp_u = torch.exp(u)
        # Limit range of logit
        exp_u = torch.clamp(exp_u, min=1e-5)

        # Moving sum:
        # Zero-pad (chunk size - 1) on the left + 1D conv with filters of 1s.
        # [batch_size, sequence_length]
        denominators = moving_sum(exp_u,
                                  back=self.chunk_size - 1, forward=0)

        # Compute beta (MoChA weights)
        beta = exp_u * moving_sum(alpha / denominators,
                                  back=0, forward=self.chunk_size - 1)
        return beta

    def hard(self, monotonic_attention, chunk_energy):
        """
        Mask non-attended area with '-inf'
        Args:
            monotonic_attention [batch_size, sequence_length]
            chunk_energy [batch_size, sequence_length]
        Return:
            masked_energy [batch_size, sequence_length]
        """
        batch_size, sequence_length = monotonic_attention.size()

        mask = monotonic_attention.new_tensor(monotonic_attention)
        for i in range(1, self.chunk_size):
            mask[:, :-i] += monotonic_attention[:, i:]

        # mask '-inf' energy before softmax
        masked_energy = chunk_energy.masked_fill_(
            (1 - mask).byte(), -float('inf'))
        return masked_energy

    @overrides
    def forward(self, q, k, v):
        """
        Soft monotonic chunkwise attention (Train)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_alpha [batch_size, sequence_length]
        Return:
            alpha [batch_size, sequence_length]
            beta [batch_size, sequence_length]
        Hard monotonic chunkwise attention (Test)
        Args:
            encoder_outputs [batch_size, sequence_length, enc_dim]
            decoder_h [batch_size, dec_dim]
            previous_attention [batch_size, sequence_length]
        Return:
            monotonic_attention [batch_size, sequence_length]: hard alpha
            chunkwise_attention [batch_size, sequence_length]: hard beta
        """
        mode = "soft" if self.training else "hard"

        chunk_energy = self.chunk_energy(q, k, v)
        monotonic_attention = self._monotonic_attention(q, k, v, mode=mode)
        if mode == "soft":
            chunkwise_attention = self.my_soft(
                monotonic_attention, chunk_energy)

        elif mode == "hard":
            masked_energy = self.hard(
                monotonic_attention, chunk_energy)
            chunkwise_attention = F.softmax(masked_energy, dim=-1)
            chunkwise_attention.masked_fill_(
                chunkwise_attention != chunkwise_attention,
                0)  # a trick to replace nan value with 0
        output = torch.bmm(chunkwise_attention, v).squeeze(1)
        return output, chunkwise_attention


class MILk(MonotonicAttention):
    def __init__(self,
                 enc_dim: int,
                 dec_dim: int,
                 att_dim: int) -> None:
        """
        [Monotonic Chunkwise Attention] from
        "Monotonic Chunkwise Attention" (ICLR 2018)
        https://openreview.net/forum?id=Hko85plCW
        """
        super().__init__(enc_dim, dec_dim, att_dim)
        self.chunk_energy = Energy(enc_dim, dec_dim, att_dim)

    @overrides
    def forward(self, encoder_outputs, decoder_h, previous_attention=None, mode="soft"):
        if mode not in ["soft", "hard"]:
            raise ValueError("Invalid forward mode {} for attention; \
                accept only soft and hard mode".format(mode))
        if mode == "soft":

            alpha = super()(encoder_outputs, decoder_h, previous_attention, mode="soft")
            chunk_energy = self.chunk_energy(encoder_outputs, decoder_h)
            beta = self.soft(alpha, chunk_energy)
            return alpha, beta

        elif mode == "hard":
            monotonic_attention = super()(encoder_outputs, decoder_h,
                                          previous_attention, mode="hard")
            chunk_energy = self.chunk_energy(encoder_outputs, decoder_h)
            masked_energy = self.hard(
                monotonic_attention, chunk_energy)
            chunkwise_attention = F.softmax(masked_energy, dim=-1)
            chunkwise_attention.masked_fill_(
                chunkwise_attention != chunkwise_attention,
                0)  # a trick to replace nan value with 0
            return monotonic_attention, chunkwise_attention

    def soft(self, emit_probs, chunk_energy):
        """
        PyTorch version of MILk in author's TF Implementation:
        https://github.com/craffel/mocha/blob/master/Demo.ipynb.
        Compute chunkwise attention distribution stably by subtracting logit max.
        """
        chunk_energy = torch.exp(chunk_energy)
        cumulative_energy = torch.cumsum(chunk_energy, dim=-1)
        return chunk_energy * fliped_cumsum(
            emit_probs / cumulative_energy, dim=-1)

    def hard(self, monotonic_attention, chunk_energy):
        """
        Mask non-attended area with '-inf'
        Args:
            monotonic_attention [batch_size, sequence_length]
            chunk_energy [batch_size, sequence_length]
        Return:
            masked_energy [batch_size, sequence_length]
        """
        batch_size, sequence_length = monotonic_attention.size()

        mask = fliped_cumsum(monotonic_attention)

        # mask '-inf' energy before softmax
        masked_energy = chunk_energy.masked_fill_(
            (1 - mask).byte(), -float('inf'))
        return masked_energy
