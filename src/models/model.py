from typing import Any, List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, PearsonCorrCoef, SpearmanCorrCoef, R2Score


class MainNet(LightningModule):
    """Main default network"""
    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5
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
        
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        
        # for logging best so far validation accuracy
        self.val_spearman_best = MaxMetric()
        self.val_pearson_best = MaxMetric()
        self.val_r2_best = MaxMetric()
        
    def forward(self, fwd_x, rev_x):        
        return self.net(fwd_x, rev_x)
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_spearman_best.reset()
        self.val_pearson_best.reset()
        self.val_r2_best.reset()
    
    def step(self, batch):
        fwd_x, rev_x, y = batch
        preds = self(fwd_x, rev_x)
        loss = self.criterion(preds, y)
        
        return loss, preds, y
    
    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        spearman = self.train_spearman(preds, targets)
        pearson = self.train_pearson(preds, targets)
        r2 = self.train_r2(preds, targets)
        metrics = {"train/loss": loss, "train/spearman": spearman, "train/pearson": pearson, "train/r2": r2}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def training_step_end(self, outputs):
        self.train_spearman.reset()
        self.train_pearson.reset()
        self.train_r2.reset()
    
    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        spearman = self.val_spearman(preds, targets)
        pearson = self.val_pearson(preds, targets)
        r2 = self.val_r2(preds, targets)
        metrics = {"val/loss": loss, "val/spearman": spearman, "val/pearson": pearson, "val/r2": r2}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
    def validation_epoch_end(self, outputs):
        # get val metric from current epoch
        spearman = self.val_spearman.compute()
        pearson = self.val_pearson.compute()
        r2 = self.val_r2.compute()
        
        # log best metric
        self.val_spearman_best.update(spearman)
        self.val_pearson_best.update(pearson)
        self.val_r2_best.update(r2)
        self.log("val/spearman_best", self.val_spearman_best.compute(), on_epoch=True, prog_bar=True)
        self.log("val/pearson_best", self.val_pearson_best.compute(), on_epoch=True, prog_bar=True)
        self.log("val/r2_best", self.val_r2_best.compute(), on_epoch=True, prog_bar=True)
        
        # reset val metrics
        self.val_spearman.reset()
        self.val_pearson.reset()
        self.val_r2.reset()
        
    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        spearman = self.test_spearman(preds, targets)
        pearson = self.test_pearson(preds, targets)
        metrics = {"test/loss": loss, "test/spearman": spearman, "test/pearson": pearson}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 
                                lr=self.hparams.lr, 
                                weight_decay=self.hparams.weight_decay)
    
    
    
class ConjoinedNet(MainNet):
    """Post-hoc conjoined setting"""
    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        super().__init__(net, lr, weight_decay)
    
    def forward(self, fwd_x, rev_x):     
        return self.net(fwd_x), self.net(rev_x)
    
    def step(self, batch):
        fwd_x, rev_x, y = batch
        fwd_pred, rev_pred = self(fwd_x, rev_x)
        fwd_loss = self.criterion(fwd_pred, y)
        rev_loss = self.criterion(rev_pred, y)
        loss = (fwd_loss + rev_loss) / 2
        pred = (fwd_pred + rev_pred) / 2
        
        return loss, pred, y
