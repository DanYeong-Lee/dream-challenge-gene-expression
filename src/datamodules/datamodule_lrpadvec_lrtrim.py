from typing import Union, Optional, Tuple
import pandas as pd
from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from src.datamodules.components.dataset_lrpadvec_lrtrim import OneHotDataset, ShiftDataset


    
class MyDataModule(LightningDataModule):
    def __init__(
        self, 
        train_dir: str = "/data/project/ddp/data/dream/train_sequences.txt",
        test_dir: str = "/data/project/ddp/data/nat2022/HQ_testdata.txt",
        predict_dir: str = "/data/project/ddp/data/dream/test_sequences.txt",  
        batch_size: int = 1024, 
        num_workers: int = 4,
        fold: Union[int, str] = "None",
        shift: bool = False,
        one_hot: bool = True,
        normalize: bool = True
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
    
    def setup(self, stage=None):
        if stage == "fit" or stage == None:
            if self.hparams.fold != "None":
                df = pd.read_csv(self.hparams.train_dir, sep="\t", names=["seq", "target"])
                if self.hparams.normalize:
                    df["target"] = (df.target - 11) / 2
                kfold = KFold(n_splits=5, shuffle=True, random_state=123456789)
                for i, (train_idx, val_idx) in enumerate(kfold.split(df)):
                    if i == self.hparams.fold:
                        break
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]
            else:
                train_df = pd.read_csv(self.hparams.train_dir, sep="\t", names=["seq", "target"])
                val_df = pd.read_csv(self.hparams.test_dir, sep="\t", names=["seq", "target"])
                if self.hparams.normalize:
                    train_df["target"] = (train_df.target - 11) / 2
                    val_df["target"] = (val_df.target - 11) / 2
            self.train_data = self.dataset(train_df)
            self.val_data = self.dataset(val_df)
        
        if stage == "test" or stage == None:
            test_df = pd.read_csv(self.hparams.test_dir, sep="\t", names=["seq", "target"])
            if self.hparams.normalize:
                test_df["target"] = (test_df.target - 11) / 2
                
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


class MyDataModule_v2(LightningDataModule):
    def __init__(
        self, 
        train_dir: str = "/data/project/ddp/data/dream/train_sequences.txt",
        test_dir: str = "/data/project/ddp/data/nat2022/HQ_testdata.txt",
        predict_dir: str = "/data/project/ddp/data/dream/test_sequences.txt",  
        batch_size: int = 1024, 
        num_workers: int = 4,
        fold: Union[int, str] = "None",
        normalize: bool = True
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
        self.dataset = OneHotDataset_v2
    
    def setup(self, stage=None):
        if stage == "fit" or stage == None:
            if self.hparams.fold != "None":
                df = pd.read_csv(self.hparams.train_dir, sep="\t", names=["seq", "target"])
                if self.hparams.normalize:
                    df["target"] = (df.target - 11) / 2
                kfold = KFold(n_splits=5, shuffle=True, random_state=123456789)
                for i, (train_idx, val_idx) in enumerate(kfold.split(df)):
                    if i == self.hparams.fold:
                        break
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]
            else:
                train_df = pd.read_csv(self.hparams.train_dir, sep="\t", names=["seq", "target"])
                val_df = pd.read_csv(self.hparams.test_dir, sep="\t", names=["seq", "target"])
                if self.hparams.normalize:
                    train_df["target"] = (train_df.target - 11) / 2
                    val_df["target"] = (val_df.target - 11) / 2
            self.train_data = self.dataset(train_df)
            self.val_data = self.dataset(val_df)
        
        if stage == "test" or stage == None:
            test_df = pd.read_csv(self.hparams.test_dir, sep="\t", names=["seq", "target"])
            if self.hparams.normalize:
                test_df["target"] = (test_df.target - 11) / 2
                
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



class NlessDataModule(MyDataModule):
    def __init__(
        self, 
        train_dir: str = "/data/project/ddp/data/dream/train_sequences.txt", 
        test_dir: str = "/data/project/ddp/data/nat2022/HQ_testdata.txt",
        predict_dir: str = "/data/project/ddp/data/dream/test_sequences.txt",  
        batch_size: int = 1024, 
        num_workers: int = 4,
        fold: Union[int, str] = "None",
        shift: bool = False,
        one_hot: bool = True,
        normalize: bool = True
    ):
        super().__init__(train_dir, test_dir, batch_size, num_workers, fold, shift, one_hot, normalize)
        
    
    def setup(self, stage=None):
        if stage == "fit" or stage == None:
            if self.hparams.fold != "None":
                df = pd.read_csv(self.hparams.train_dir, sep="\t", names=["seq", "target"])
                df = df[df.seq.map(lambda x: "N" not in x)]  # No N in sequence
                if self.hparams.normalize:
                    df["target"] = (df.target - 11) / 2
                kfold = KFold(n_splits=5, shuffle=True, random_state=123456789)
                for i, (train_idx, val_idx) in enumerate(kfold.split(df)):
                    if i == self.hparams.fold:
                        break
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]
            else:
                train_df = pd.read_csv(self.hparams.train_dir, sep="\t", names=["seq", "target"])
                train_df = train_df[train_df.seq.map(lambda x: "N" not in x)]  # No N in sequence
                val_df = pd.read_csv(self.hparams.test_dir, sep="\t", names=["seq", "target"])
                if self.hparams.normalize:
                    train_df["target"] = (train_df.target - 11) / 2
                    val_df["target"] = (val_df.target - 11) / 2
            self.train_data = self.dataset(train_df)
            self.val_data = self.dataset(val_df)
        
        if stage == "test" or stage == None:
            test_df = pd.read_csv(self.hparams.test_dir, sep="\t", names=["seq", "target"])
            if self.hparams.normalize:
                test_df["target"] = (test_df.target - 11) / 2
                
            self.test_data = self.dataset(test_df)
            
        if stage == "predict" or stage == None:
            predict_df = pd.read_csv(self.hparams.predict_dir, sep="\t", names=["seq", "target"])
            self.predict_data = self.dataset(predict_df)
    