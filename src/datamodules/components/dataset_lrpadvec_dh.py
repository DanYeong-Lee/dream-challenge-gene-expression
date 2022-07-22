import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import numpy as np
import random


class ShiftDataset(Dataset):
    def __init__(self, df, max_length=110, tta=3):
        super().__init__()
        self.base2int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.one_hot_matrix = torch.tensor([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [0., 0., 0., 0.],
        ])
        
        self.records = df.to_records()
        self.max_length = max_length

        self.tta = tta
        assert self.tta % 2 == 1

    def _get_shifted_sequence(self, seq, shift):
        if shift == 0:
            return seq
        elif shift < 0:
            shift = -shift
            return 'CGATTCGAAC'[-shift:] + seq[:-shift]
        else:
            return seq[shift:] + 'TCTTAATTAA'[:shift] # 10bp scaffold + seq + 10bp scaffold

    def _pad(self, seq):
        # Make sure that sequence length is exactly `max_length`.
        vector_left = 'GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAAC'
        vector_right = 'TCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACGA'
        
        if len(seq) < self.max_length:
            if random.random() < 0.5: 
                seq = seq + vector_right[:self.max_length - len(seq)] # Pad right.
            else:
                seq = vector_left[-(self.max_length - len(seq)):] + seq # Pad left.
        elif len(seq) > self.max_length: # Trim it from the right.
            seq = seq[:self.max_length]

        # if len(seq) < self.max_length:
            # seq = 'N' * (self.max_length - len(seq)) + seq  # Pad left with 'N'.
        # elif len(seq) > self.max_length:
            # seq = seq[-self.max_length:]  # Trim it from the left.

        return seq
    
    def __getitem__(self, i):
        seq, exp = self.records[i].seq, self.records[i].target

        # Target standardization.
        exp = (exp - 11.0) / 2.0

        seqs = []
        shift_range = [i - self.tta // 2 for i in range(self.tta)]
        for shift in shift_range:
            seq_shifted = self._get_shifted_sequence(seq, shift)
            seq_shifted = self._pad(seq_shifted)

            seq_shifted_int = [self.base2int[base] for base in seq_shifted]
            seq_shifted = self.one_hot_matrix[seq_shifted_int].T
            seqs.append(seq_shifted)

            seq_shifted_rc = seq_shifted.flip([0, 1])
            seqs.append(seq_shifted_rc)
        
        seqs.append(torch.tensor([exp]).float())
        
        return tuple(seqs)

    def __len__(self):
        return len(self.records)


class BaseDataset(Dataset):
    def __init__(self, df, max_length=110):
        super().__init__()
        self.base2int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.one_hot_matrix = torch.tensor([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [0., 0., 0., 0.],
        ])
        
        self.records = df.to_records()
        self.max_length = max_length
    
    def __getitem__(self, i):
        seq, exp = self.records[i].seq, self.records[i].target

        # Target standardization.
        exp = (exp - 11.0) / 2.0

        # Make sure that sequence length is exactly `max_length`.
        vector_left = 'GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAAC'
        vector_right = 'TCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACGA'
        
        if len(seq) < self.max_length:
            if random.random() < 0.5: 
                seq = seq + vector_right[:self.max_length - len(seq)] # Pad right.
            else:
                seq = vector_left[-(self.max_length - len(seq)):] + seq # Pad left.
        elif len(seq) > self.max_length: # Trim it from the right.
            seq = seq[:self.max_length]

        # One-hot encode sequence.
        seq_int = [self.base2int[base] for base in seq]
        seq = self.one_hot_matrix[seq_int].T # Produces 4 x max_length one-hot tensor. (in fact it's one-hot except columns for 'N')

        seq_rc = seq.flip([0, 1])

        return seq, seq_rc, torch.tensor([exp]).float()

    def __len__(self):
        return len(self.records)