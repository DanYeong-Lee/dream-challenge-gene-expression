import numpy as np
import torch
from torch.utils.data import Dataset
from Bio.Seq import Seq
import itertools


class KmerDataset(Dataset):
    def __init__(
        self, 
        df,
        k
    ):
        self.records = df.to_records()

        self.k = k
        kmers = ["".join(v) for v in itertools.product(*["ATCGN"] * k)]
        kmer2idx = {kmer: i for i, kmer in enumerate(kmers)}


        def idx2vec(length, idx):
            vec = [0.] * length
            vec[idx] += 1

            return vec

        self.kmer2vec = {kmer: idx2vec(5**k, idx) for idx, kmer in enumerate(kmer2idx)}
    
    def seq2mat(self, seq, max_len=110):
        if len(seq) > max_len:
            seq = seq[len(seq) - max_len :]
        else:
            seq = "N" * (max_len - len(seq)) + seq
        
        mat = []
        for i in range(len(seq) - self.k + 1):
            mat.append(self.kmer2vec[seq[i:i+self.k]])

        return torch.tensor(mat, dtype=torch.float32)
    
    def reverse_complement(self, fwd_tensor):
        temp = fwd_tensor.flip(0)
        rev_tensor = temp.index_select(dim=1, index=torch.LongTensor([1, 0, 3, 2]))
        
        return rev_tensor

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        _, seq, target = self.records[idx]
        X = self.seq2mat(seq)
        X_rev = self.seq2mat(Seq(seq).reverse_complement())
        y = torch.tensor(float(target), dtype=torch.float32)
        
        return X, X_rev, y
    
    
class ShiftDataset(Dataset):
    def __init__(
        self, 
        df
    ):
        self.records = df.to_records()
        self.base2vec = {
            "A": [1., 0., 0., 0.],
            "T": [0., 1., 0., 0.],
            "C": [0., 0., 1., 0.],
            "G": [0., 0., 0., 1.],
            "N": [0., 0., 0., 0.]
        }
    
    def seq2mat(self, seq, max_len=110):
        if len(seq) > max_len:
            seq = seq[len(seq) - max_len :]
        else:
            seq = "N" * (max_len - len(seq)) + seq
        mat = torch.tensor(list(map(lambda x: self.base2vec[x], seq)), dtype=torch.float32)
        return mat
    
    def reverse_complement(self, fwd_tensor):
        temp = fwd_tensor.flip(0)
        rev_tensor = temp.index_select(dim=1, index=torch.LongTensor([1, 0, 3, 2]))
        
        return rev_tensor
        
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        _, seq, target = self.records[idx]
        ls_seq = "C" + seq[:-1]
        rs_seq = seq[1:] + "T"
        fwd_tensors = [self.seq2mat(ls_seq), self.seq2mat(seq), self.seq2mat(rs_seq)]
        rev_tensors = [self.reverse_complement(fwd_tensor) for fwd_tensor in fwd_tensors]
        tensors = fwd_tensors + rev_tensors
        y = torch.tensor(float(target), dtype=torch.float32)
        tensors.append(y)
        
        return tuple(tensors)
    
