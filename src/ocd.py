import torch
import torch.nn as nn

def lev(ref, hyp):
    table = [[0 for i,_ in enumerate(ref)] for j,_ in enumerate(hyp)]
    for i, _ in enumerate(hyp):
        table[i][0] = i
    for j, _ in enumerate(ref):
        table[0][j] = j
    for i, h in enumerate(hyp):
        for j, r in enumerate(ref):
            if i < 1 or j < 1:
                continue
            table[i][j] = min(table[i-1][j] + 1, \
                    table[i][j-1] + 1, table[i-1][j-1] + (h != r))
    return table


def ocd(ref):
    table = [list(range(len(ref)))]
    def step(h):
        i = len(table)
        if h != None:
            row = [i]
            for j, r in enumerate(ref[:-1]):
                j += 1
                val = min(table[i-1][j] + 1, \
                        row[j-1] + 1, table[i-1][j-1] + (h != r))
                row.append(val)
            table.append(row)
        min_d = min(table[-1])
        cands_idx = [idx for idx, d in enumerate(table[-1]) if d == min_d]
        cands = [ref[i] for i in cands_idx]
        return cands, min_d

    return step

def gen_policy(out_probs, samples, labels, temp=1e-8):
    temp = max(temp, 1e-8)
    q_vals = []
    for out_prob, sample, label in zip(out_probs, samples, labels):
        sample = sample.tolist()
        label = label.tolist()
        q_val = torch.zeros_like(out_prob)
        play = ocd(label)
        for t, char in enumerate([None] + sample[:-1]):
            cands, m = play(char)
            cands = torch.LongTensor(cands)
            q_val[t] = -(m + 1)
            q_val[t,cands] = -m

        q_vals.append(q_val)

    q_vals = torch.stack(q_vals, dim=0)
    policies = nn.functional.softmax(q_vals / temp, dim=-1)

    return policies


