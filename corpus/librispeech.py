from tqdm import tqdm
from pathlib import Path
from os.path import join, getsize
from joblib import Parallel, delayed
from torch.utils.data import Dataset

from src.util import mp_progress_map

# Additional (official) text src provided
OFFICIAL_TXT_SRC = ['librispeech-lm-norm.txt']
# Remove longest N sentence in librispeech-lm-norm.txt
REMOVE_TOP_N_TXT = 5000000
# Default num. of threads used for loading LibriSpeech
READ_FILE_THREADS = 4



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
    def __init__(self, path, split, tokenizer, bucket_size, ascending=False, wave_to_feat=None):
        # Setup
        self.path = path
        self.bucket_size = bucket_size

        # List all wave files
        file_list, spkr_id_list = [], []
        self.spkr_id_dict = {}
        for s in split:
            file_list += list(Path(join(path, s)).rglob("*.flac"))
            spkr_id_list += sorted([int(item.name)
                                    for item in Path(join(path, s)).iterdir() if item.is_dir()])
        assert len(file_list) > 0, "No data found @ {}".format(path)

        # Generate speaker id dict
        spkr_id_list = list(dict.fromkeys(spkr_id_list))  # Remove duplicate id
        for idx, spkr_id in enumerate(spkr_id_list):
            self.spkr_id_dict[spkr_id] = idx
        self.spkr_num = len(self.spkr_id_dict)

        # Read text
        text = Parallel(n_jobs=READ_FILE_THREADS)(
            delayed(read_text)(str(f)) for f in file_list)
        #text = Parallel(n_jobs=-1)(delayed(tokenizer.encode)(txt) for txt in text)
        text = [tokenizer.encode(txt) for txt in text]
        # get indices that would sort an array
        indices = sorted(range(len(text)), reverse=not ascending, key=lambda idx: len(text.__getitem__(idx)))
        # Sort dataset by text length
        #file_len = Parallel(n_jobs=READ_FILE_THREADS)(delayed(getsize)(f) for f in file_list)
        self.file_list = [file_list[idx] for idx in indices]
        self.text = [text[idx] for idx in indices]
        # Process wavefiles to features
        self.features = None
        if callable(wave_to_feat):
            #self.features = Parallel(n_jobs=READ_FILE_THREADS)(delayed(wave_to_feat)(f) for f in file_list)
            self.features = mp_progress_map(wave_to_feat,
                               ((f,) for f in file_list), READ_FILE_THREADS)
        # self.file_list, self.text = zip(*[(f_name, txt)
        #                                   for f_name, txt in sorted(zip(file_list, text),
        #                                   reverse=not ascending, key=lambda x:len(x[1]))])

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_list)-self.bucket_size, index)
            return [(self.file_list[idx] if self.features is None else self.features[idx],
                     self.text[idx], self.get_id(self.file_list[idx])) for idx in
                    range(index, index + self.bucket_size)]
                    #zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]
        else:
            f_path = self.file_list[index]
            feat = f_path if self.features is None else self.features[index]
            return feat, self.text[index], self.get_id(f_path)

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
