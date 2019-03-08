import editdistance as ed
import pandas as pd
import argparse

# Arguments
parser = argparse.ArgumentParser(description='Evaluate decoding result.')
parser.add_argument('--file', type=str, help='Path to decode result file.')
paras = parser.parse_args()

                        
with open(paras.file) as f:
    pairs = f.read().splitlines()
    truth, pred = list(zip(*[pair.split('\t') for pair in pairs]))
cer = []
wer = []
for gt,pd in zip(truth,pred):
    wer.append(ed.eval(pd.split(' '),gt.split(' '))/len(gt.split(' ')))
    cer.append(ed.eval(pd,gt)/len(gt))

print('CER : {:.6f}'.format(sum(cer)/len(cer)))
print('WER : {:.6f}'.format(sum(wer)/len(wer)))
print('p.s. for phoneme sequences, WER=Phone Error Rate and CER is meaningless.')
