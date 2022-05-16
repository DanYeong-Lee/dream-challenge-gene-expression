from typing import Optional, Tuple

from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from src.datamodules.components.dataset import OneHotDataset, IndexDataset, get_len


    
class MyDataModule(LightningDataModule):
    def __init__(
        self, 
        train_dir: str = "/data/project/ddp/data/dream/train_sequences.txt", 
        test_dir: str ="/data/project/ddp/data/dream/test_sequences.txt",  
        batch_size: int = 1024, 
        num_workers: int = 4,
        fold: int = 0,
        one_hot: bool = True
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
            
        if one_hot:
            self.dataset = OneHotDataset
        else:
            self.dataset = IndexDataset

    def setup(self, stage=None):
        if stage == "fit" or stage == None:
            train_length = get_len(self.hparams.train_dir)
            kfold = KFold(n_splits=5, shuffle=True, random_state=123456789)
            for i, (train_idx, val_idx) in enumerate(kfold.split(range(train_length))):
                if i == self.hparams.fold:
                    break

            self.train_data = self.dataset(self.hparams.train_dir, train_idx)
            self.val_data = self.dataset(self.hparams.train_dir, val_idx)
        
        if stage == "test" or stage == None:
            test_length = get_len(self.hparams.test_dir)
            self.test_data = self.dataset(self.hparams.test_dir, range(test_length))
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers)
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers)
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers)

