from typing import Union, Optional, Tuple
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from src.datamodules.components.dataset import OneHotDataset, IndexDataset, ShiftDataset, OneHotDataset_v2


    
class UniformDataModule(LightningDataModule):
    def __init__(
        self, 
        train_dir: str = "/data/project/ddp/data/dream/train_sequences.txt",
        test_dir: str = "/data/project/ddp/data/nat2022/HQ_testdata.txt",
        predict_dir: str = "/data/project/ddp/data/dream/test_sequences.txt",  
        batch_size: int = 1024, 
        num_workers: int = 4,
        fold: Union[int, str] = "None",
        shift: bool = False,
        one_hot: bool = True
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
        
        if self.hparams.shift:
            self.dataset = ShiftDataset
        elif self.hparams.one_hot:
            self.dataset = OneHotDataset
        else:
            self.dataset = IndexDataset
    
    def setup(self, stage=None):
        if stage == "fit" or stage == None:
            if self.hparams.fold != "None":
                df = pd.read_csv(self.hparams.train_dir, sep="\t", names=["seq", "target"])
                df["target"] = stats.norm.cdf(df.target, loc=11.0, scale=2.0) * 3 - 1.5
                kfold = KFold(n_splits=5, shuffle=True, random_state=123456789)
                for i, (train_idx, val_idx) in enumerate(kfold.split(df)):
                    if i == self.hparams.fold:
                        break
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]
            else:
                train_df = pd.read_csv(self.hparams.train_dir, sep="\t", names=["seq", "target"])
                val_df = pd.read_csv(self.hparams.test_dir, sep="\t", names=["seq", "target"])
                train_df["target"] = stats.norm.cdf(train_df.target, loc=11.0, scale=2.0) * 3 - 1.5
                val_df["target"] = stats.norm.cdf(val_df.target, loc=11.0, scale=2.0) * 3 - 1.5
            self.train_data = self.dataset(train_df)
            self.val_data = self.dataset(val_df)
        
        if stage == "test" or stage == None:
            test_df = pd.read_csv(self.hparams.test_dir, sep="\t", names=["seq", "target"])
            test_df["target"] = stats.norm.cdf(test_df.target, loc=11.0, scale=2.0) * 3 - 1.5
                
            self.test_data = self.dataset(test_df)
            
        if stage == "predict" or stage == None:
            predict_df = pd.read_csv(self.hparams.predict_dir, sep="\t", names=["seq", "target"])
            self.predict_data = self.dataset(predict_df)
    
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
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )

