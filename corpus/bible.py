from tqdm import tqdm
import json
from pathlib import Path
from os.path import join,getsize
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from util.tl2phone import tailo2phone_factory

OFFICIAL_TXT_SRC = ['librispeech-lm-norm.txt']
READ_FILE_THREADS = 4

def read_text(file):
    '''Get transcription of target wave file, 
       it's somewhat redundant for accessing each txt multiplt times,
       but it works fine with multi-thread'''
    src_file = '-'.join(file.split('-')[:-1])+'.trans.txt'
    idx = file.split('/')[-1].split('.')[0]

    with open(src_file,'r') as fp:
        for line in fp:
            if idx == line.split(' ')[0]:
                return line[:-1].split(' ',1)[1]

class TSMBibleDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        root = Path(path)

        # get transcriptions
        with open(root.joinpath('data.json')) as f:
            transcriptions = {utt["mp3"]: utt["tailo"] for utt in json.load(f)}

        # List all wave files
        wav_files = []
        trns = []
        for s in split:
            split_path = root.joinpath(s)
            for wavfile in split_path.rglob("*.mp3"):
                try:
                    trns.append(transcriptions[str(wavfile.relative_to(split_path))])
                    wav_files.append(wavfile)
                except KeyError:
                    print("wav file {} not found; skipping".format(wavfile))
        
        # Read text
        tailo2phone = tailo2phone_factory()
        trns = [tokenizer.encode(tailo2phone(trn)) for trn in trns]
        
        # Read file size and sort dataset by file size (Note: feature len. may be different)
        file_len = Parallel(n_jobs=READ_FILE_THREADS)(delayed(getsize)(f) for f in wav_files)
        self.file_list, self.text = zip(*[(f_name,txt) \
                    for _,f_name,txt in sorted(zip(file_len,wav_files,trns), reverse=True, key=lambda x:x[0])])

    def __getitem__(self,index):
        if self.bucket_size>1:
            # Return a bucket
            index = min(len(self.file_list)-self.bucket_size,index)
            return [(f_path, txt) for f_path,txt in \
                     zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]
        else:
            return self.file_list[index], self.text[index]


    def __len__(self):
        return len(self.file_list)

