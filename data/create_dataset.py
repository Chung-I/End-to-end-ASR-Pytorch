import sys
import os
import lmdb # install lmdb by "pip install lmdb"
import librosa
import numpy as np
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed
import random
import argparse
import json
from tsm_utils import poj_tl, tl2phone
import re
from collections import Counter
from functools import reduce


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def gen_dict(outputPath, dataList):
    sents = list(map(lambda x: x[1], dataList))
    tokens = reduce(lambda x,y: x + y, [sent.split() for sent in sents])
    typs = Counter(tokens)
    id2word = ['<sos>', '<eos>', '<unk>']
    id2word += list(map(lambda x: x[0], typs.most_common()))
    word2id = {word: idx for idx, word in enumerate(id2word)}
    print(word2id)
    with open(outputPath, 'wb') as fp:
        pickle.dump(word2id, fp)


def getDataList(inputPath):
    dataList = []
    for drama in os.listdir(inputPath):
        with open(os.path.join(inputPath, drama, 'caption')) as f:
            labels = f.read().splitlines()
        audioFiles = os.listdir(os.path.join(inputPath, drama, 'segment'))
        audioFiles = sorted(audioFiles, key=lambda x: int(x.split('.')[0]))
        audioFiles = list(map(lambda f: os.path.join(inputPath, drama, 'segment', f), audioFiles))
        pairs = list(zip(audioFiles, labels))
        dataList += pairs
    dataList = sorted(dataList, key=lambda pair: len(pair[1]))
    return dataList


def getDataListFromJson(inputPath):
    dataList = []
    json_file = os.path.join(inputPath, 'data.json')
    with open(json_file) as f:
        datas = json.load(f)
    for data in datas:
        poj = data['tailo']
        tls = poj_tl(poj).pojt_tlt().tlt_tls()
        tls = ' '.join(re.findall('\w+', tls.lower()))
        try:
            phones = [tl2phone(tl) for tl in re.split('[\s\-]+', tls)]
            sent = ' '.join(phones)
        except Exception as e:
            print(tls, data['mp3'])
            continue
        audioFile = os.path.join(inputPath, data['mp3'])
        dataList.append((audioFile, sent))
    dataList = sorted(dataList, key=lambda pair: len(pair[1]))
    return dataList


def extract_feature(input_file,feature='fbank',dim=40, cmvn=True, delta=False, delta_delta=False,
                    window_size=25, stride=10,save_feature=None, speed=1.0):
    try:
        y, sr = librosa.load(input_file,sr=None)
    except:
        print('{} not found, skipping'.format(input_file))
        return None
    duration = librosa.get_duration(y=y, sr=sr)
    if duration < 0.15:
        return None

    y = librosa.to_mono(y.T)
    if sr != 16000:
        y = librosa.resample(y, sr, 16000)
        sr = 16000
    if speed != 1.0:
        y = librosa.core.resample(y, sr, sr * speed)
    ws = int(sr*0.001*window_size)
    st = int(sr*0.001*stride)
    if feature == 'fbank': # log-scaled
        feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=dim,
                                    n_fft=ws, hop_length=st)
        #feat = 10 * np.log10(np.maximum(1e-10, feat))
    elif feature == 'mfcc':
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=dim, n_mels=26,
                                    n_fft=ws, hop_length=st)
        feat[0] = librosa.feature.rmse(y, hop_length=st, frame_length=ws) 
        
    else:
        raise ValueError('Unsupported Acoustic Feature: '+feature)

    feat = [feat]
    if delta:
        feat.append(librosa.feature.delta(feat[0]))

    if delta_delta:
        feat.append(librosa.feature.delta(feat[0],order=2))
    feat = np.concatenate(feat,axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:,np.newaxis]) / (feat.std(axis=1)+1e-16)[:,np.newaxis]
    if save_feature is not None:
        tmp = np.swapaxes(feat,0,1).astype('float32')
        np.save(save_feature,tmp)
        return len(tmp)
    else:
        return np.swapaxes(feat,0,1).astype('float32')


def wrapper(audio_file, label):
    feat = extract_feature(audio_file, dim=80)
    return feat, label, audio_file


def createDataset(outputPath, dataList, num_thd=1): 
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        checkValid    : if true, check the validity of every image
    """
    def save_to_lmdb(cache, feat, label, full_path):
        audioKey = 'audio-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        fileKey = 'file-%09d' % cnt
        paths = full_path.split('/')
        drama = paths[-3]
        fileName = paths[-1]
        fileName = drama + '-' + fileName
        cache[audioKey] = feat.tobytes()
        cache[labelKey] = label.encode()
        cache[fileKey] = fileName.encode()

    nSamples = len(dataList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 0
    step = 1000
    if num_thd > 1:
        for i in tqdm(range(0, nSamples, step)):
            feats = Parallel(n_jobs=4)(delayed(wrapper)(*data) for data in dataList[i:i+step])

            for feat, label, full_path in feats:
                if feat is None:
                    continue
                save_to_lmdb(cache, feat, label, full_path)
                cnt += 1
            writeCache(env, cache)
            del cache
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
    else:
        for i in tqdm(range(0, nSamples)):
            #feats = Parallel(n_jobs=4)(delayed(wrapper)(*data) for data in dataList[i:i+step])
            full_path, label = dataList[i]
            try:
                feat = extract_feature(full_path, dim=80)
            except Exception as e:
                print(e)
                continue
            if feat is None:
                continue
            save_to_lmdb(cache, feat, label, full_path)
            cnt += 1
            if cnt % 1000 == 0:
                writeCache(env, cache)
                del cache
                cache = {}
                print('Written %d / %d' % (cnt, nSamples))
    nSamples = cnt
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out')
    parser.add_argument('--inf')
    parser.add_argument('--gen_dict', action='store_true')
    parser.add_argument('--threads', type=int, default=1, help='number of threads.')
    parser.add_argument('--format', choices=['pair', 'path', 'json'], default='pair',
            help='format of input path')
    args = parser.parse_args()
    outputPath = args.out
    inputPath = args.inf
    if args.format == 'pair':
        with open(os.path.join(inputPath, 'src-{}.txt'.format(sys.argv[3]))) as f:
            srcs = [os.path.join(inputPath, line) for line in f.read().splitlines()]
        with open(os.path.join(inputPath, 'tgt-{}.txt'.format(sys.argv[3]))) as f:
            tgts = f.read().splitlines()
        dataList = list(zip(srcs, tgts))
        random.shuffle(dataList)
        dataList = sorted(dataList, key=lambda pair: len(pair[1]))
    elif args.format == 'path':
        dataList = getDataList(inputPath)
    elif args.format == 'json':
        dataList = getDataListFromJson(inputPath)

    if args.gen_dict:
        gen_dict(outputPath, dataList)
    else:
        createDataset(outputPath, dataList, args.threads)
