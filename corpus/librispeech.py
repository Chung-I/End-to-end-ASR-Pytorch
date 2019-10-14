from typing import Callable

from tqdm import tqdm
from pathlib import Path
from os.path import join, getsize
from joblib import Parallel, delayed
import numpy as np
import torchaudio

import torch
from torch.utils.data import Dataset

from src.util import mp_progress_map, wave_to_feat_and_save_factory, write_sliced_array

# Additional (official) text src provided
OFFICIAL_TXT_SRC = ['librispeech-lm-norm.txt']
# Remove longest N sentence in librispeech-lm-norm.txt
REMOVE_TOP_N_TXT = 5000000
# Default num. of threads used for loading LibriSpeech
READ_FILE_THREADS = 12


def read_text(file):
    '''Get transcription of target wave file,
       it's somewhat redundant for accessing each txt multiplt times,
       but it works fine with multi-thread'''
    src_file = '-'.join(file.split('-')[:-1])+'.trans.txt'
    idx = file.split('/')[-1].split('.')[0]

    with open(src_file, 'r') as fp:
        for line in fp:
            if idx == line.split(' ')[0]:
                return line[:-1].split(' ', 1)[1]


class LibriDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size, ascending=False, wave_to_feat=None, in_memory=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size

        # List all wave files

        # Process wavefiles to features
        self.features = None
        self.aug_features = None
        self.waves = None
        list_of_features = []
        list_of_feat_lens = []
        list_of_aug_features = []
        list_of_aug_feat_lens = []
        file_list = []

        if not in_memory:
            for s in split:
                file_list += list(Path(join(path, s)).rglob("*.flac"))
        elif in_memory == 'wave':
            for s in split:
                file_list += list(Path(join(path, s)).rglob("*.flac"))
            self.waves, _ = zip(*mp_progress_map(torchaudio.load, ((f,)
                                                                   for f in file_list), READ_FILE_THREADS))
        elif in_memory == True or in_memory == 'mmap':
            def pt_path_to_np_array(path): return torch.load(path).numpy()
            mmap_mode = 'r' if in_memory == 'mmap' else None
            for s in split:
                split_dir = Path(join(path, s))
                data_file = split_dir.joinpath('data.npy')
                aug_data_file = split_dir.joinpath('aug_data.npy')
                if not data_file.exists():
                    files = list(split_dir.rglob("*.flac"))
                    with open(data_file.with_name("files.txt"), 'w') as fp:
                        fp.write(
                            "\n".join([str(f.relative_to(split_dir)) for f in files]))

                    feat_lens, feat_names, aug_feat_lens, aug_feat_names = \
                        zip(*mp_progress_map(wave_to_feat_and_save_factory(wave_to_feat),
                                             ((f,) for f in files), READ_FILE_THREADS))

                    write_sliced_array(feat_names, str(data_file), sum(feat_lens),
                                       func=pt_path_to_np_array)
                    np.save(data_file.with_name(
                        "lens.npy"), np.array(feat_lens))

                    if aug_feat_lens[0] is not None:
                        write_sliced_array(aug_feat_names, str(aug_data_file),
                                           sum(aug_feat_lens), func=pt_path_to_np_array)
                        np.save(data_file.with_name("aug_lens.npy"),
                                np.array(aug_feat_lens))
                else:
                    with open(data_file.with_name("files.txt")) as fp:
                        files = [split_dir.joinpath(line)
                                 for line in fp.read().splitlines()]
                    print(f"feature file {data_file} exists; loading")
                list_of_features.append(torch.from_numpy(
                    np.load(data_file, mmap_mode=mmap_mode)))
                feat_lens = np.load(data_file.with_name("lens.npy"))
                list_of_feat_lens.append(feat_lens)

                if aug_data_file.exists():
                    print(
                        f"augmented feature file {aug_data_file} exists; loading")
                    list_of_aug_features.append(torch.from_numpy(
                        np.load(aug_data_file, mmap_mode=mmap_mode)))
                    aug_feat_lens = np.load(
                        aug_data_file.with_name("aug_lens.npy"))
                    list_of_aug_feat_lens.append(aug_feat_lens)

                file_list += files
        else:
            raise NotImplementedError

            self.features = torch.cat(list_of_features, dim=0) \
                if len(list_of_features) > 1 else list_of_features[0]
            self.aug_features = torch.cat(list_of_aug_features, dim=0) \
                if len(list_of_aug_features) > 1 else list_of_aug_features[0]
            feat_ptr = np.pad(np.concatenate(list_of_feat_lens, axis=0), (1, 0), mode='constant').cumsum() \
                if list_of_feat_lens else None
            aug_feat_ptr = np.pad(np.concatenate(list_of_aug_feat_lens, axis=0), (1, 0), mode='constant').cumsum() \
                if list_of_aug_feat_lens else None

        assert len(file_list) > 0, "No data found @ {}".format(path)

        # Read text
        text = Parallel(n_jobs=READ_FILE_THREADS)(
            delayed(read_text)(str(f)) for f in file_list)
        text = [tokenizer.encode(txt) for txt in text]

        # Generate speaker id dict
        self.spkr_id_dict = {}
        spkr_id_list = []
        for s in split:
            spkr_id_list += sorted([int(item.name)
                                    for item in Path(join(path, s)).iterdir() if item.is_dir()])
        spkr_id_list = list(dict.fromkeys(spkr_id_list))  # Remove duplicate id
        for idx, spkr_id in enumerate(spkr_id_list):
            self.spkr_id_dict[spkr_id] = idx
        self.spkr_num = len(self.spkr_id_dict)
        self.spkr_id_list = spkr_id_list

        spkr_id_count = [0] * self.spkr_num
        for f in file_list:
            idx = self.get_id(f)
            spkr_id_count[idx] += 1
        spkr_count_reci = 1/torch.FloatTensor(spkr_id_count)
        self.spkr_weight = spkr_count_reci/spkr_count_reci.sum()
        # Speaker things done.

        indices = sorted(range(len(text)), reverse=not ascending,
                         key=lambda idx: len(text.__getitem__(idx)))
        self.file_list = [file_list[idx] for idx in indices]
        self.text = [text[idx] for idx in indices]
        if self.features is not None:
            feat_intvls = [(feat_ptr[idx], feat_ptr[idx+1]) for idx in indices]
        if self.aug_features is not None:
            aug_feat_intvls = [(aug_feat_ptr[idx], aug_feat_ptr[idx+1])
                               for idx in indices]

        if self.waves is not None:
            self.waves = [self.waves[idx] for idx in indices]
            self.get_feat = lambda idx: self.waves[idx]
        elif self.features is None:
            self.get_feat = lambda idx: self.file_list[idx]
        elif self.aug_features is None:
            self.get_feat = lambda idx: (
                self.features[feat_intvls[idx][0]:feat_intvls[idx][1]],)
        else:
            self.get_feat = lambda idx: (self.features[feat_intvls[idx][0]:feat_intvls[idx][1]],
                                         self.aug_features[aug_feat_intvls[idx][0]:aug_feat_intvls[idx][1]])

    def __getitem__(self, index):

        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_list)-self.bucket_size, index)
            return [(self.get_feat(idx), self.text[idx], self.get_id(self.file_list[idx])) for idx in
                    range(index, index + self.bucket_size)]
            # zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]
        else:
            return self.get_feat(index), self.text[index], self.get_id(self.file_list[index])

    def __len__(self):
        return len(self.file_list)

    def get_id(self, file):
        return self.spkr_id_dict[int(file.name.split('-')[0])]


class LibriTextDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        self.encode_on_fly = False
        read_txt_src = []

        # List all wave files
        file_list, all_sent = [], []

        for s in split:
            if s in OFFICIAL_TXT_SRC:
                self.encode_on_fly = True
                with open(join(path, s), 'r') as f:
                    all_sent += f.readlines()
            file_list += list(Path(join(path, s)).rglob("*.flac"))
        assert (len(file_list) > 0) or (len(all_sent)
                                        > 0), "No data found @ {}".format(path)

        # Read text
        text = Parallel(n_jobs=READ_FILE_THREADS)(
            delayed(read_text)(str(f)) for f in file_list)
        all_sent.extend(text)
        del text

        # Encode text
        if self.encode_on_fly:
            self.tokenizer = tokenizer
            self.text = all_sent
        else:
            self.text = [tokenizer.encode(txt) for txt in tqdm(all_sent)]
        del all_sent

        # Read file size and sort dataset by file size (Note: feature len. may be different)
        self.text = sorted(self.text, reverse=True, key=lambda x: len(x))
        if self.encode_on_fly:
            del self.text[:REMOVE_TOP_N_TXT]

    def __getitem__(self, index):
        if self.bucket_size > 1:
            index = min(len(self.text)-self.bucket_size, index)
            if self.encode_on_fly:
                for i in range(index, index+self.bucket_size):
                    if type(self.text[i]) is str:
                        self.text[i] = self.tokenizer.encode(self.text[i])
            # Return a bucket
            return self.text[index:index+self.bucket_size]
        else:
            if self.encode_on_fly and type(self.text[index]) is str:
                self.text[index] = self.tokenizer.encode(self.text[index])
            return self.text[index]

    def __len__(self):
        return len(self.text)
