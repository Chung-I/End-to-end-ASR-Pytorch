from typing import Dict
from pathlib import Path
from collections import namedtuple

import torch
import torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal

import pandas as pd
from lib.filters import create_mel_filterbank
from lib.mfcc import create_mfcc_transform
from src.util import mp_progress_map

import librosa
import random


GFL_ITER = 30  # iteration of griffin lim
MIN_LEVEL_DB = -100
REF_LEVEL_DB = 20
MFCC_HOP_LEN_MS = 10
MFCC_WIN_LEN_MS = 25
N_MFCC_NO_DELTA = 13
NoiseSource = namedtuple('NoiseSource', ['files', 'snr_range', 'n_files_range'])

class CMVN(torch.jit.ScriptModule):

    __constants__ = ["mode", "dim", "eps"]

    def __init__(self, mode="global", dim=2, eps=1e-10):
        # `torchaudio.load()` loads audio with shape [channel, feature_dim, time]
        # so perform normalization on dim=2 by default
        super(CMVN, self).__init__()

        if mode != "global":
            raise NotImplementedError(
                "Only support global mean variance normalization.")

        self.mode = mode
        self.dim = dim
        self.eps = eps

    @torch.jit.script_method
    def forward(self, x):
        if self.mode == "global":
            return (x - x.mean(self.dim, keepdim=True)) / (self.eps + x.std(self.dim, keepdim=True))

    def extra_repr(self):
        return "mode={}, dim={}, eps={}".format(self.mode, self.dim, self.eps)


class Delta(torch.jit.ScriptModule):

    __constants__ = ["order", "window_size", "padding"]

    def __init__(self, order=1, window_size=2):
        # Reference:
        # https://kaldi-asr.org/doc/feature-functions_8cc_source.html
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_audio.py
        super(Delta, self).__init__()

        self.order = order
        self.window_size = window_size

        filters = self._create_filters(order, window_size)
        self.register_buffer("filters", filters)
        self.padding = (0, (filters.shape[-1] - 1) // 2)

    @torch.jit.script_method
    def forward(self, x):
        # Unsqueeze batch dim
        return F.conv2d(x, weight=self.filters, padding=self.padding)

    # TODO(WindQAQ): find more elegant way to create `scales`
    def _create_filters(self, order, window_size):
        scales = [[1.0]]
        for i in range(1, order + 1):
            prev_offset = (len(scales[i-1]) - 1) // 2
            curr_offset = prev_offset + window_size

            curr = [0] * (len(scales[i-1]) + 2 * window_size)
            normalizer = 0.0
            for j in range(-window_size, window_size + 1):
                normalizer += j * j
                for k in range(-prev_offset, prev_offset + 1):
                    curr[j+k+curr_offset] += (j * scales[i-1][k+prev_offset])
            curr = [x / normalizer for x in curr]
            scales.append(curr)

        max_len = len(scales[-1])
        for i, scale in enumerate(scales[:-1]):
            padding = (max_len - len(scale)) // 2
            scales[i] = [0] * padding + scale + [0] * padding

        return torch.Tensor(scales).unsqueeze(1).unsqueeze(1)

    def extra_repr(self):
        return "order={}, window_size={}".format(self.order, self.window_size)


class Postprocess(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, x):
        # [channel, feature_dim, time] -> [time, channel, feature_dim]
        x = x.permute(2, 0, 1)
        # [time, channel, feature_dim] -> [time, feature_dim * channel]
        return x.reshape(x.size(0), -1).detach()


# TODO(Windqaq): make this scriptable
class ExtractAudioFeature(nn.Module):
    def __init__(self, mode="fbank", num_mel_bins=40, **kwargs):
        super(ExtractAudioFeature, self).__init__()
        self.mode = mode
        self.extract_fn = torchaudio.compliance.kaldi.fbank if mode == "fbank" else torchaudio.compliance.kaldi.mfcc
        self.num_mel_bins = num_mel_bins
        self.kwargs = kwargs

    def forward(self, filepath):
        waveform, sample_rate = torchaudio.load(filepath)

        y = self.extract_fn(waveform,
                            num_mel_bins=self.num_mel_bins,
                            channel=-1,
                            sample_frequency=sample_rate,
                            **self.kwargs)
        return y.transpose(0, 1).unsqueeze(0).detach()

    def extra_repr(self):
        return "mode={}, num_mel_bins={}".format(self.mode, self.num_mel_bins)


def create_transform(audio_config):
    feat_type = audio_config.pop("feat_type")
    feat_dim = audio_config.pop("feat_dim")

    delta_order = audio_config.pop("delta_order", 0)
    delta_window_size = audio_config.pop("delta_window_size", 2)
    apply_cmvn = audio_config.pop("apply_cmvn")

    transforms = [ExtractAudioFeature(feat_type, feat_dim, **audio_config)]

    if delta_order >= 1:
        transforms.append(Delta(delta_order, delta_window_size))

    if apply_cmvn:
        transforms.append(CMVN())

    transforms.append(Postprocess())

    return nn.Sequential(*transforms), feat_dim * (delta_order + 1)


class AudioProcessor(nn.Module):
    def __init__(self, num_freq, num_mels, frame_shift_ms, frame_length_ms,
                 preemphasis_coeff, sample_rate, **kwargs):
        super(AudioProcessor, self).__init__()
        self.n_fft = (num_freq - 1) * 2
        self.n_mels = num_mels
        self.num_freq = num_freq
        self.hop_length = int(frame_shift_ms / 1000 * sample_rate)
        self.win_length = int(frame_length_ms / 1000 * sample_rate)
        self.hop_length_mfcc = int(MFCC_HOP_LEN_MS / 1000 * sample_rate)
        self.win_length_mfcc = int(MFCC_WIN_LEN_MS / 1000 * sample_rate)
        self.window = torch.hann_window(self.win_length)
        self.preemphasis_coeff = preemphasis_coeff
        self.sr = sample_rate
        self.to_specgram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            # [NOTICE]
            # What we want is power=1, but this is a HACK for the bug of torchaudio's spectrogram (power=1)
            power=2
        )
        self.to_specgram_mfcc = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length_mfcc,
            hop_length=self.hop_length_mfcc,
            # [NOTICE]
            # What we want is power=1, but this is a HACK for the bug of torchaudio's spectrogram (power=1)
            power=2
        )
        self.to_melspecgram = torchaudio.transforms.MelScale(
            n_mels=self.n_mels,
            sample_rate=self.sr
        )

        _mel_basis = create_mel_filterbank(
            self.sr, self.n_fft, n_mels=self.n_mels).T
        _mel_basis = torch.from_numpy(_mel_basis)
        # HACK : torchaudio only creates f-banks after first pass
        # tmp_fb = torchaudio.functional.create_fb_matrix(self.num_freq, self.to_melspecgram.f_min,
        #                                                self.to_melspecgram.f_max, self.to_melspecgram.n_mels)

        self.to_melspecgram.fb.resize_(_mel_basis.size())
        self.to_melspecgram.fb.copy_(_mel_basis)

    def load(self, wav_path):
        """
        Return:
            waveform of shape (samples)
        """
        if isinstance(wav_path, str) or isinstance(wav_path, Path):
            waveform, sr = torchaudio.load(wav_path)
            assert self.sr == sr, "Sample rate mismatch. Expected %d but get %d" \
                % (self.sr, sr)
        elif isinstance(wav_path, torch.Tensor):
            waveform = wav_path
        else:
            raise NotImplementedError
        return waveform

    def extract_feature_from_file(self, wav_path, preemphasis=True, channel=0):
        """
        Arg:
            wav_path: file path
            preemphasis: preemphasize or not (High-pass filter)
            channel: channel to extract
        Return:
            specgram: spectrogram of shape (freq, time)
            melspecgram: mel spectrogram of shape (n_mels, time)
        """
        waveform = self.load(wav_path)
        return self.extract_feature_from_waveform(
            waveform, preemphasis=preemphasis, channel=channel)

    def segment(self, feat, boundary):
        """
        Arg:
            mfcc: torch.Tensor of shape (time, mfcc dim)
            boundary: phone boundaries in terms of ratio over time axis
        Return:
            mfcc: segments of MFCC　of shape (segment, time, mffc dim)
        """

        feat_len, feat_dim = feat.shape
        # Calculate all segment point and make sure each segment>1
        segment_point = []
        prev_sp, max_sg_len = 0, 0
        for sp in [round(b*feat_len) for b in boundary]:
            sg_len = sp-prev_sp
            max_sg_len = max(max_sg_len, sg_len)
            if sg_len >= self.min_segment_len:
                segment_point.append((prev_sp, sp))
                prev_sp = sp
        # Slice mfcc into SxTxD
        segment_feat = torch.zeros((len(segment_point), max_sg_len, feat_dim))
        for i, (l, r) in enumerate(segment_point):
            segment_feat[i, :(r-l), :] = feat[l:r, :]
        return segment_feat

    def extract_mfcc_from_file(self, wav_path, preemphasis=True, channel=0):
        """
        Arg:
            wav_path: file path
            preemphasis: preemphasize or not (High-pass filter)
            channel: channel to extract
        Return:
            segmented mfcc: mfcc of shape (# segment, len of segment, 39)
        """
        waveform = self.load(wav_path)
        mfcc = self.extract_mfcc_from_waveform(waveform, preemphasis, channel)
        return mfcc

    def extract_mfcc_from_waveform(self, waveform, preemphasis=True, channel=0):
        """
        Arg:
            waveform: torch.Tensor of shape (channels, samples)
            preemphasis: preemphasize or not (High-pass filter)
            channel: channel to extract
        Return:
            mfcc: mfcc of shape (freq, time)
        """
        with torch.no_grad():
            if preemphasis:
                waveform = self._preemphasis(waveform)
            # sqrt(): HACK for the bug of torchaudio's spectrogram (power=1)
            specgram = self.to_specgram_mfcc(waveform).sqrt()
            melspecgram = self.to_melspecgram(specgram)
            melspecgram = self._amp_to_db(melspecgram) - REF_LEVEL_DB
            melspecgram = self._normalize(melspecgram)

            # TODO: use torch to calculate it
            mfcc = librosa.feature.mfcc(
                S=melspecgram[channel].numpy(), n_mfcc=N_MFCC_NO_DELTA)
            mfcc_features = [mfcc, librosa.feature.delta(
                mfcc), librosa.feature.delta(mfcc, order=2)]
            mfcc = torch.from_numpy(np.concatenate(mfcc_features, axis=0))
            return mfcc

    def extract_feature_from_waveform(self, waveform, preemphasis=True, channel=0):
        """
        Arg:
            waveform: torch.Tensor of shape (channels, samples)
            preemphasis: preemphasize or not (High-pass filter)
            channel: channel to extract
        Return:
            specgram: spectrogram of shape (freq, time)
            melspecgram: mel spectrogram of shape (n_mels, time)
        """
        with torch.no_grad():
            if preemphasis:
                waveform = self._preemphasis(waveform)
            # sqrt(): HACK for the bug of torchaudio's spectrogram (power=1)
            specgram = self.to_specgram(waveform).sqrt()
            melspecgram = self.to_melspecgram(specgram)
            specgram = self._amp_to_db(specgram) - REF_LEVEL_DB
            specgram = self._normalize(specgram)
            melspecgram = self._amp_to_db(melspecgram) - REF_LEVEL_DB
            melspecgram = self._normalize(melspecgram)

        return specgram[channel], melspecgram[channel]

    def specgram_to_waveform(self, specgram, power=1.0, inv_preemphasis=True, isAmp=False):
        """
        Arg:
            specgram: torch.Tensor of shape (freq, time)
        Return:
            approximate waveform: numpy array of shape (samples)
        """
        if not isAmp:
            specgram = self._denormalize(specgram)
            specgram = self._db_to_amp(specgram + REF_LEVEL_DB) ** power
        wav = self._griffin_lim(specgram).detach().cpu().numpy()
        if inv_preemphasis:
            wav = self._inv_preemphasis(wav)
        return np.clip(wav, -1, 1)

    def melspecgram_to_specgram(self, melspecgram):
        """
        Arg:
            melspecgram: torch.Tensor of shape (freq[mel], time)
        Return:
            approximate spectrogram: numpy array of shape (freq[spectrogram], time)
        """
        # (freq[mel], )
        fb_pinv = torch.pinverse(self.to_melspecgram.fb).transpose(0, 1)
        melspecgram = self._db_to_amp(
            self._denormalize(melspecgram) + REF_LEVEL_DB)
        specgram = torch.matmul(fb_pinv, melspecgram)
        return specgram

    def _griffin_lim(self, specgram):
        """
        Arg:
            specgram: torch.Tensor of shape (freq, time)
        Return:
            approximate waveform of shape (samples)
        """
        phases = np.angle(np.exp(2j * np.pi * np.random.rand(*specgram.shape)))
        phases = phases.astype(np.float32)
        phases = torch.from_numpy(phases)
        magnitude = specgram.abs()
        # Spectrum with random phases
        y = self._to_complex(magnitude, phases)
        x = self._istft(y)
        for _ in range(GFL_ITER):
            y = self._stft(x)
            phases = self._get_phase(y)
            y = self._to_complex(magnitude, phases)
            x = self._istft(y)
        return x

    def _preemphasis(self, waveform):
        waveform = torch.cat([
            waveform[:, :1],
            waveform[:, 1:] - self.preemphasis_coeff * waveform[:, :-1]], dim=-1)
        return waveform

    def _stft(self, x):
        # `x` for time-domain signal and `y` for frequency-domain signal
        y = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True)
        return y

    def _istft(self, y):
        # `x` for time-domain signal and `y` for frequency-domain signal
        x = torchaudio.functional.istft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True)
        return x

    def _to_complex(self, magnitude, phase):
        """To make a fake complex number in torch"""
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        complx = torch.stack([real, imag], dim=-1)
        return complx

    def _get_phase(self, complx):
        return torchaudio.functional.angle(complx)

    def _inv_preemphasis(self, wav):
        """Note this is implemented in 'scipy' but not 'torch'!!"""
        return scipy.signal.lfilter([1], [1, -0.97], wav)

    def _amp_to_db(self, x, minimum=1e-5):
        return 20 * torch.log10(torch.clamp(x, min=minimum))

    def _db_to_amp(self, x):
        return 10 ** (0.05 * x)

    def _normalize(self, feat):
        return torch.clamp((feat - MIN_LEVEL_DB) / -MIN_LEVEL_DB, min=0, max=1)

    def _denormalize(self, feat):
        return MIN_LEVEL_DB + torch.clamp(feat, min=0, max=1) * -MIN_LEVEL_DB


class AudioConverter(AudioProcessor):
    """A wrapper of AudioProcessor"""

    def __init__(self, num_freq, num_mels, frame_length_ms, frame_shift_ms, preemphasis_coeff,
                 sample_rate, use_linear, noise, snr_range, time_stretch_range, inverse_prob,
                 segment_file, segment_feat, min_segment_len, in_memory):
        super(AudioConverter, self).__init__(
            num_freq, num_mels, frame_shift_ms, frame_length_ms,
            preemphasis_coeff, sample_rate)
        self.use_linear = use_linear
        self.noise_root = Path(noise['path'])
        self.noise_sources: Dict[str, NoiseSource] = {}
        if noise.get('genre') is not None:
            for noise_type, (_snr_range, n_files_range) in noise['genre'].items():
                files = list(self.noise_root.joinpath(noise_type).rglob("*.wav"))
                if in_memory == 'wave':
                    files, _ = zip(*mp_progress_map(torchaudio.load, ((f,) for f in files), 6))
                noise_source = NoiseSource(files, _snr_range, n_files_range)
                self.noise_sources[noise_type] = noise_source
        self.snr_range = snr_range
        self.time_stretch_range = time_stretch_range
        self.inverse_prob = inverse_prob
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms
        # Feature information
        self.feat_type = 'Mel/Linear' if use_linear else 'Mel'
        self.feat_dim = (num_mels, num_freq) if use_linear else (
            num_mels, None)

        # Mfcc slicer
        self.use_segment = segment_file is not None
        if self.use_segment:
            self.segment_src = segment_file
            self.segment_feat = segment_feat.lower()
            self.min_segment_len = min_segment_len
            seg_feat_dim = None
            if self.segment_feat == 'mfcc':
                seg_feat_dim = 39  # TODO: to config?
            elif self.segment_feat == 'mel':
                seg_feat_dim = num_mels
            elif self.segment_feat == 'linear':
                seg_feat_dim = num_freq
            else:
                raise NotImplementedError
            self.seg_feat_dim = seg_feat_dim
            #self.mfcc_extractor, self.mfcc_dim = create_mfcc_transform(self.sr)
            self.boundary_table = pd.read_csv(
                segment_file, index_col=0) if self.use_segment else None
            self.boundary_table['seg'] = [compute_len_ratio(
                v) for v in self.boundary_table['seg'].values]

    def wave_to_feat(self, file):
        # -- old -- #
        # sp, msp = self.extract_feature_from_waveform(wave)
        if isinstance(file, Path) or isinstance(file, str):
            file = str(file)
            wave = self.load(file)
        elif isinstance(file, torch.Tensor):
            wave = file

        _sp, _msp = self.extract_feature_from_waveform(wave)
        # _mfcc = self.extract_mfcc_from_file(file)
        # Reshape (T, D)
        sp = _sp.T if self.use_linear else None
        msp = _msp.T

        seg_feat = None
        if self.use_segment:
            # Get boundary
            file_key = str(file).split('/')[-1].split('.')[0]
            boundary = self.boundary_table.loc[file_key]['seg']
            # -------- Segment --------
            if self.segment_feat == 'mfcc':
                seg_feat = _mfcc
            elif self.segment_feat == 'mel':
                seg_feat = _msp
            elif self.segment_feat == 'linear':
                seg_feat = _sp
            else:
                raise NotImplementedError

        # Augmentation
        apply_noise = -1 not in self.snr_range
        apply_real_noise = len(self.noise_sources) > 0
        (f"apply_real_noise: {apply_real_noise}")
        apply_time_stretch = not (
            self.time_stretch_range[0] == self.time_stretch_range[1] == 1)
        msp_aug = msp.clone()
        if apply_noise or apply_real_noise or apply_time_stretch:
            with torch.no_grad():
                wave_aug = wave.clone()
                # 1. Add noise
                if apply_noise:
                    snr = random.uniform(self.snr_range[0], self.snr_range[1])
                    wave_aug = self.add_noise(wave_aug, snr)

                if apply_real_noise:
                    noise_type = random.choice(list(self.noise_sources.keys()))
                    noise_files, snr_range, n_files_range = self.noise_sources[noise_type]
                    n_files = random.randint(*n_files_range)
                    sampled_files = random.sample(noise_files, n_files)
                    noises = []
                    for noise_file in sampled_files:
                        noise = self.load(noise_file)
                        noises.append(noise)

                    snr = random.uniform(*snr_range)
                    wave_aug = self.add_real_noise(wave_aug, noises, snr)
                    # save as wavfile for debugging
                    # torchaudio.save(str(Path("tmp").joinpath(noise_files[0].name)), wave_aug, self.sr)

                # 2. Time stretch
                if apply_time_stretch:
                    stretch_rate = 1
                else:
                    stretch_rate = random.uniform(
                        self.time_stretch_range[0], self.time_stretch_range[1])

                sr = int(self.sr * stretch_rate)
                wave_aug = self._preemphasis(wave_aug)
                sp_aug = self._spectrogram(
                    waveform=wave_aug,
                    n_fft=self.n_fft,
                    win_length=int(self.frame_length_ms / 1000 * sr),
                    hop_length=int(self.frame_shift_ms / 1000 * sr),
                    power=2).sqrt()
                # To mel spectrogram
                fb = create_mel_filterbank(
                    self.sr, self.n_fft, n_mels=self.n_mels).T
                fb = torch.from_numpy(fb)

                msp_aug = torch.matmul(
                    sp_aug.transpose(1, 2), fb).transpose(1, 2)

                msp_aug = self._amp_to_db(msp_aug) - REF_LEVEL_DB
                msp_aug = self._normalize(msp_aug)
                msp_aug = msp_aug[0].T  # 1st channel
            return (msp, msp_aug)
        else:
            return (msp,)

    def feat_to_wave(self, feat):
        # (D, T)
        feat = feat.transpose(0, 1)
        isAmp = False
        if feat.size(0) == self.feat_dim[0]:
            # feat is melspecgram
            isAmp = True
            feat = self.melspecgram_to_specgram(feat)
        # Spectrogram -> wave
        wave = self.specgram_to_waveform(feat, isAmp=isAmp)
        return wave, self.sr

    def add_noise(self, signal, snr):
        for ch, s in enumerate(signal):
            noise = torch.randn(s.shape[0])
            coeff = snr_coeff(snr, s, noise)
            signal[ch] = s + (coeff * noise)
        return signal

    def add_real_noise(self, signal, noises, snr):
        signal_len = signal.shape[1]
        aggregated_noise = torch.zeros(signal_len)

        for noise in noises:
            assert noise.shape[0] == 1
            noise = noise[0]

            noise_len = noise.shape[0]

            if signal_len <= noise_len:
                start = random.randint(0, noise_len - signal_len)
                modified_noise = noise[start:start + signal_len]
            else:
                n_repeats = signal_len // noise_len + 1
                modified_noise = noise.repeat(n_repeats)
                modified_noise = modified_noise[:signal_len]

            aggregated_noise += modified_noise

        for ch, s in enumerate(signal):
            coeff = snr_coeff(snr, s, aggregated_noise)
            signal[ch] = s + (coeff * aggregated_noise)
        return signal

    def _spectrogram(self, waveform, n_fft, win_length, hop_length, pad=0,
                     window_fn=torch.hann_window, power=2, normalized=False):
        window = window_fn(win_length)
        return torchaudio.functional.spectrogram(
            waveform, pad, window, n_fft, hop_length, win_length, power, normalized)


def compute_len_ratio(v):
    '''
        Return the boundaries in terms of ratio over time axis
        Note: this is an approximation, not exactly the boundary found by MFA
    '''
    tmp = list(map(float, v.split('_')))
    max_len = tmp[-1]
    return [t/max_len for t in tmp]


def snr_coeff(snr, signal, noise):
    pwr_signal = (signal ** 2).sum().item()
    pwr_noise = (noise ** 2).sum().item()
    return (pwr_signal / pwr_noise * 10 ** (-snr / 10)) ** 0.5


def load_audio_transform(num_freq, num_mels, frame_length_ms, frame_shift_ms,
                         preemphasis_coeff, sample_rate, use_linear, noise, snr_range, time_stretch_range,
                         inverse_prob, segment_file=None, segment_feat=None, min_segment_len=2, in_memory=False):
    ''' Return a audio converter specified by config '''

    audio_converter = AudioConverter(num_freq, num_mels, frame_length_ms, frame_shift_ms,
                                     preemphasis_coeff, sample_rate, use_linear, noise, snr_range,
                                     time_stretch_range, inverse_prob, segment_file, segment_feat,
                                     min_segment_len, in_memory)

    return audio_converter
