from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from src.datamodules.components.dataset import OneHotDataset, IndexDataset


    
class MyDataModule(LightningDataModule):
    def __init__(
        self, 
        train_dir: str = "/data/project/ddp/data/dream/train_sequences.txt", 
        test_dir: str ="/data/project/ddp/data/dream/test_sequences.txt",  
        batch_size: int = 1024, 
        num_workers: int = 4,
        fold: int = 0,
        one_hot: bool = True,
        normalize: bool = True
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
            
        self.normalize = True

    def setup(self, stage=None):
        if stage == "fit" or stage == None:
            df = pd.read_csv(self.hparams.train_dir, sep="\t", names=["seq", "target"])
            if self.normalize:
                df["target"] = (df.target - 11) / 2
            kfold = KFold(n_splits=5, shuffle=True, random_state=123456789)
            for i, (train_idx, val_idx) in enumerate(kfold.split(df)):
                if i == self.hparams.fold:
                    break
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            self.train_data = self.dataset(train_df)
            self.val_data = self.dataset(val_df)
        
        if stage == "test" or stage == None:
            test_df = pd.read_csv(self.hparams.test_dir, sep="\t", names=["seq", "target"])
            self.test_data = self.dataset(test_df)
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )

