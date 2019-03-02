from torch.utils.data import Sampler
import torch
import numpy as np

class BatchBucketSampler(Sampler):

    def __init__(self, size, bucket_size, batch_size, drop_last):
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.n = size
        self.b = batch_size
        self.w = bucket_size
        self.drop_last = drop_last

    def __iter__(self):
        batches = []
        buckets = []
        start = 0
        starts = list(range(0, self.n, self.w))
        starts = [starts[i] for i in torch.randperm(len(starts)).tolist()]
        
        for start in starts:
            max_idx = self.n - (self.n - start) % self.b  if self.drop_last else self.n
            end = min(start + self.w, max_idx)
            bucket = [start + i for i in torch.randperm(end-start).tolist()]
            s = 0
            while s < len(bucket):
                batches.append(bucket[s:s+self.b])
                s += self.b

        return (batches[i] for i in torch.randperm(len(batches)).tolist())

    def __len__(self):
        if self.drop_last:
            return self.n//self.b
        else:
            return int(np.ceil(self.n/self.b))
