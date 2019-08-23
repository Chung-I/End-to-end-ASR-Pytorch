from tqdm import tqdm
from pathlib import Path
from os.path import join,getsize
from joblib import Parallel, delayed
from torch.utils.data import Dataset
import re

OFFICIAL_TXT_SRC = ['librispeech-lm-norm.txt']
READ_FILE_THREADS = 4

def process_phone(phone, remove_tone=True):
    if remove_tone:
        phone = re.sub("\d+", "", phone)
    return phone

def word_to_phone_factory(lexicon):
    def w2p(word):
        phones = []
        try:
            phones.extend(re.split("\s+", lexicon[word]))
        except KeyError:
            for char in word:
                try:
                    phones.extend(re.split("\s+", lexicon[char]))
                except KeyError:
                    pass
        phones = [process_phone(phone) for phone in phones]
        return phones

    return w2p

def get_lexicon(lexicon_path):
    lexicon = {}
    with open(lexicon_path) as fp:
        for line in fp:
            ws = line.index(" ")
            word, phoneme = line[:ws], line[ws+1:]
            lexicon[word] = phoneme
    return lexicon

def read_text(path: Path):
    '''Get transcription of target wave file, 
       it's somewhat redundant for accessing each txt multiplt times,
       but it works fine with multi-thread'''
    src_file = path.parents[1].joinpath('txt').joinpath(path.stem + ".txt")

    with open(src_file,'r') as fp:
        return fp.read()

class PTSDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        lexicon_path = "/home/nlpmaster/lexicon.txt"
        lexicon = get_lexicon(lexicon_path)
        w2p = word_to_phone_factory(lexicon)
        # List all wave files
        file_list = []
        for s in split:
            file_list += list(Path(join(path,s)).rglob("*.wav"))
        
        # Read text
        text = Parallel(n_jobs=READ_FILE_THREADS)(delayed(read_text)(f) for f in file_list)
        #text = Parallel(n_jobs=-1)(delayed(tokenizer.encode)(txt) for txt in text)
        text = [" ".join([" ".join(w2p(word)) for word in transcript.split()]) for transcript in text]
        indices = [tokenizer.encode(re.sub("\s+", " ", txt)) for txt in text]
        
        # Read file size and sort dataset by file size (Note: feature len. may be different)
        file_len = Parallel(n_jobs=READ_FILE_THREADS)(delayed(getsize)(f) for f in file_list)
        self.file_list, self.text = zip(*[(f_name,txt) \
                    for _,f_name,txt in sorted(zip(file_len,file_list, indices), reverse=True, key=lambda x:x[0])])

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

    @staticmethod
    def get_transcripts(path, split):

        # List all wave files
        file_list = []
        for s in split:
            file_list += list(Path(join(path,s)).rglob("*.wav"))
        
        # Read text
        text = Parallel(n_jobs=READ_FILE_THREADS)(delayed(read_text)(f) for f in file_list)

        return text
        

if __name__ == "__main__":
    import re
    from collections import Counter

    flatten = lambda l: [item for sublist in l for item in sublist]


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default="/home/nlpmaster/ssd-1t/corpus/PTS-MSub-Vol1")
    parser.add_argument('--splits', nargs='+', default=['segmented'])
    parser.add_argument('--lexicon', default="/home/nlpmaster/lexicon.txt")
    parser.add_argument('--vocab', default="tests/sample_data/pts.vocab")
    args = parser.parse_args()

    lexicon = get_lexicon(args.lexicon)
    w2p = word_to_phone_factory(lexicon)
    transcripts = PTSDataset.get_transcripts(args.root, args.splits)
    phoneme_transcripts = [[w2p(word) for word in transcript.split()] for transcript in transcripts]
    phonemes = flatten(phoneme_transcripts)
    phonemes = flatten(phonemes)
    vocab = Counter(phonemes)
    with open(args.vocab, "w") as fp:
        for phoneme, _ in vocab.most_common():
            if phoneme:
                fp.write(phoneme + "\n")

