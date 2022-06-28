from typing import Any, List
import json
from collections import OrderedDict
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, PearsonCorrCoef, SpearmanCorrCoef
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


class MainNet(LightningModule):
    """Main default network"""
    """Use only forward strand"""
    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        
        self.net = net
        
        self.criterion = nn.MSELoss()
        
        self.train_pearson = PearsonCorrCoef()
        self.val_pearson = PearsonCorrCoef()
        self.test_pearson = PearsonCorrCoef()
        
        self.train_spearman = SpearmanCorrCoef()
        self.val_spearman = SpearmanCorrCoef()
        self.test_spearman = SpearmanCorrCoef()
        
        # for logging best so far validation accuracy
        self.val_spearman_best = MaxMetric()
        self.val_pearson_best = MaxMetric()
        
    def forward(self, fwd_x):        
        return self.net(fwd_x)
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_spearman_best.reset()
        self.val_pearson_best.reset()
    
    def step(self, batch):
        fwd_x, rev_x, y = batch
        preds = self(fwd_x)
        loss = self.criterion(preds, y)
        
        return loss, preds, y
    
    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        batch_spearman = self.train_spearman(preds, targets)
        batch_pearson = self.train_pearson(preds, targets)
        metrics = {"train/loss_batch": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def training_epoch_end(self, outputs):
        # get metric from current epoch
        epoch_spearman = self.train_spearman.compute()
        epoch_pearson = self.train_pearson.compute()
        
        # log epoch metrics
        metrics = {"train/spearman": epoch_spearman, "train/pearson": epoch_pearson}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        # reset metrics
        self.train_spearman.reset()
        self.train_pearson.reset()
        
    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.val_spearman.update(preds, targets)
        self.val_pearson.update(preds, targets)
        
        metrics = {"val/loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
    def validation_epoch_end(self, outputs):
        # get val metric from current epoch
        epoch_spearman = self.val_spearman.compute()
        epoch_pearson = self.val_pearson.compute()
        
        if self.val_pearson.n_total.item() > 10000:
            # CV
            # log epoch metrics
            metrics = {"val/spearman": epoch_spearman, "val/pearson": epoch_pearson}
            self.log_dict(metrics, on_epoch=True, prog_bar=True)

            # log best metric
            self.val_spearman_best.update(epoch_spearman)
            self.val_pearson_best.update(epoch_pearson)
            self.log("val/spearman_best", self.val_spearman_best.compute(), on_epoch=True, prog_bar=True)
            self.log("val/pearson_best", self.val_pearson_best.compute(), on_epoch=True, prog_bar=True)
        else:
            # Validating with HQ_testata
            metrics = {"test/full_spearman": epoch_spearman, "test/full_pearson": epoch_pearson}
            self.log_dict(metrics, on_epoch=True, prog_bar=True)

            # log best metric
            self.val_spearman_best.update(epoch_spearman)
            self.val_pearson_best.update(epoch_pearson)
            self.log("test/full_spearman_best", self.val_spearman_best.compute(), on_epoch=True, prog_bar=True)
            self.log("test/full_pearson_best", self.val_pearson_best.compute(), on_epoch=True, prog_bar=True)
            
        # reset val metrics
        self.val_spearman.reset()
        self.val_pearson.reset()
        
    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.test_spearman.update(preds, targets)
        self.test_pearson.update(preds, targets)
        metrics = {"test/loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
    def test_epoch_end(self, outputs):
        # get val metric from current epoch
        epoch_spearman = self.test_spearman.compute()
        epoch_pearson = self.test_pearson.compute()
        
        # log epoch metrics
        metrics = {"test/spearman": epoch_spearman, "test/pearson": epoch_pearson}
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        
        
    def predict_step(self, batch, batch_idx):
        _, preds, _ = self.step(batch)
        
        return preds
    
    def on_predict_epoch_end(self, outputs):
        with open("../../../../../sample_submission.json", "r") as f:
            ground = json.load(f)
    
        indices = np.array([int(indice) for indice in list(ground.keys())])
        PRED_DATA = OrderedDict()
        Y_pred = np.array(torch.cat(outputs[0]))

        for i in indices:
            PRED_DATA[str(i)] = float(Y_pred[i])
        
        
        with open("../../../../../submission.json", "w") as f:
            json.dump(PRED_DATA, f)

        print("Saved submission file!")
        
    def on_fit_end(self):
        self.trainer.save_checkpoint("last-swa.ckpt")
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 
                                lr=self.hparams.lr, 
                                weight_decay=self.hparams.weight_decay)
    

class DSNet(MainNet):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        
    ):
        super().__init__(net, lr, weight_decay)
    
    def forward(self, fwd_x, rev_x):        
        return self.net(fwd_x, rev_x)
    
    def step(self, batch):
        fwd_x, rev_x, y = batch
        preds = self(fwd_x, rev_x)
        loss = self.criterion(preds, y)
        
        return loss, preds, y
    
    
class ConjoinedNet(MainNet):
    """Post-hoc conjoined setting"""
    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        super().__init__(net, lr, weight_decay)
    
    def forward(self, tensors):     
        return [self.net(tensor) for tensor in tensors]
    
    def step(self, batch):
        Xs = batch[: -1]
        y = batch[-1]
        
        preds = self(Xs)
        losses = torch.stack([self.criterion(pred, y) for pred in preds])
        
        loss = losses.mean()
        pred = torch.stack(preds).mean(dim=0)
        
        return loss, pred, y


class ConjoinedNet_AW(ConjoinedNet):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-2
    ):
        super().__init__(net, lr, weight_decay)
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), 
                                lr=self.hparams.lr, 
                                weight_decay=self.hparams.weight_decay)
    
    
class ConjoinedNet_CAW(ConjoinedNet):
    """Post-hoc conjoined setting"""
    """+ CosineAnnealingWarmupRestarts"""
    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0,
        first_cycle_steps: int = 3,
        cycle_mult: float = 1.0,
        max_lr: float = 1e-2,
        min_lr: float = 1e-4,
        warmup_steps: int = 2,
        gamma: float = 1.0
    ):
        super().__init__(net, lr, weight_decay)
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.first_cycle_steps,
            cycle_mult=self.cycle_mult,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_steps=self.warmup_steps,
            gamma=self.gamma
        )
        
        return [optimizer], [scheduler]
    
    
class ConjoinedNet_CA(ConjoinedNet):
    """Post-hoc conjoined setting"""
    """+ CosineAnnealing"""
    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0,
        max_epochs: int = 20,
        eta_min: float = 0.0
    ):
        super().__init__(net, lr, weight_decay)
        self.max_epochs = max_epochs
        self.eta_min = eta_min
    
    def configure_optimizers(self):
        n_steps = len(self.trainer._data_connector._train_dataloader_source.dataloader())
        
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs * n_steps,
            eta_min=self.eta_min
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    
class ConjoinedNet_AW_CA(ConjoinedNet):
    """Post-hoc conjoined setting"""
    """+ CosineAnnealing"""
    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0,
        max_epochs: int = 20,
        eta_min: float = 0.0
    ):
        super().__init__(net, lr, weight_decay)
        self.max_epochs = max_epochs
        self.eta_min = eta_min
    
    def configure_optimizers(self):
        n_steps = len(self.trainer._data_connector._train_dataloader_source.dataloader())
        
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs * n_steps,
            eta_min=self.eta_min
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    
class ConjoinedNet_AW_CAW(ConjoinedNet):
    """Post-hoc conjoined setting"""
    """+ CosineAnnealingWarmupRestarts"""
    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0,
        first_cycle_steps: int = 3,
        cycle_mult: float = 1.0,
        max_lr: float = 1e-2,
        min_lr: float = 1e-4,
        warmup_steps: int = 2,
        gamma: float = 1.0
    ):
        super().__init__(net, lr, weight_decay)
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.first_cycle_steps,
            cycle_mult=self.cycle_mult,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_steps=self.warmup_steps,
            gamma=self.gamma
        )
        
        return [optimizer], [scheduler]
    
    
    
class MainNet_CA(MainNet):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0,
        max_epochs: int = 20,
        eta_min: float = 0.0
    ):
        super().__init__(net, lr, weight_decay)
        self.max_epochs = max_epochs
        self.eta_min = eta_min
    
    def configure_optimizers(self):
        n_steps = len(self.trainer._data_connector._train_dataloader_source.dataloader())
        
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs * n_steps,
            eta_min=self.eta_min
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    
class MainNet_AW_CA(MainNet):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        max_epochs: int = 20,
        eta_min: float = 0.0
    ):
        super().__init__(net, lr, weight_decay)
        self.max_epochs = max_epochs
        self.eta_min = eta_min
    
    def configure_optimizers(self):
        n_steps = len(self.trainer._data_connector._train_dataloader_source.dataloader())
        
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs * n_steps,
            eta_min=self.eta_min
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]