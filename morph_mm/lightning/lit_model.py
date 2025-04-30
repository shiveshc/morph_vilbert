import torch
import lightning as L

from ml_collections import ConfigDict

from morph_mm.models.img_bert import ImgBERT
from morph_mm.lightning.losses import L2_loss

from typing import Dict


class LitImgBert(L.LightningModule):
    def __init__(self,
                 config:ConfigDict,
                 model:ImgBERT) -> None:
        super().__init__()

        self.config = config
        self.model = ImgBERT(config)
    

    def training_step(self, batch:Dict[str, torch.Tensor]):
        pred, gt = self.model(batch)
        l = L2_loss(pred, gt)
        return l

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer