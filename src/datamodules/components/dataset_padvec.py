import numpy as np
import torch
from torch.utils.data import Dataset
from Bio.Seq import Seq


class OneHotDataset(Dataset):
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
    
    def seq2mat(self,seq, max_len=110):
        vec = "TCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACG"
        seq = seq + vec
        seq = seq[:max_len]
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
        X = self.seq2mat(seq)
        X_rev = self.reverse_complement(X)
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
        seq = seq[:max_len]
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
        seq = seq + "TCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACG"
        ls_seq = "C" + seq[:-1] + "TCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACG"
        rs_seq = seq[1:] + "TCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACG"
        fwd_tensors = [self.seq2mat(ls_seq), self.seq2mat(seq), self.seq2mat(rs_seq)]
        rev_tensors = [self.reverse_complement(fwd_tensor) for fwd_tensor in fwd_tensors]
        tensors = fwd_tensors + rev_tensors
        y = torch.tensor(float(target), dtype=torch.float32)
        tensors.append(y)
        
        return tuple(tensors)
    
