import torch
import lightning as L

from ml_collections import ConfigDict

from morph_mm.models.img_bert import ImgBERT
from morph_mm.lightning import losses

from typing import Dict


class LitImgBert(L.LightningModule):
    def __init__(self,
                 config:ConfigDict) -> None:
        super().__init__()

        self.config = config
        self.model = ImgBERT(config)
        if config.loss == 'l2':
            self.loss_fn = losses.L2_loss
        elif config.loss == 'l1':
            self.loss_fn = losses.L1_loss
        else:
            raise NotImplementedError(f'{config.loss} is not implemented')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def forward(self, batch:Dict[str, torch.Tensor]):
        pred, gt, cls_token, orig_img, recon_img = self.model(batch)
        return pred, gt, cls_token, orig_img, recon_img

    def training_step(self, batch:Dict[str, torch.Tensor]):
        pred, gt, cls_token, orig_img, recon_img = self.forward(batch)
        l = self.loss_fn(pred, gt)
        self.log("train_loss", l)
        return l
    
    def validation_step(self, batch:Dict[str, torch.Tensor]):
        pred, gt, cls_token, orig_img, recon_img = self.forward(batch)
        l = self.loss_fn(pred, gt)
        self.log("val_loss", l)
        return l

    def test_step(self, batch:Dict[str, torch.Tensor]):
        pred, gt, cls_token, orig_img, recon_img = self.forward(batch)
        l = self.loss_fn(pred, gt)
        self.log("test_loss", l)
        return l