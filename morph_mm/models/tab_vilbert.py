import torch
import torch.nn as nn
import numpy as np


class ViLBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vcoder = transformer()
        self.tcoder = transformer()

        tab_fields = config['tab_fields']
        tab_embedder = {fname: nn.Embedding(fdim) for fname, fdim in tab_fields.items()}

    
    def forward(self, batch):
        img = batch['img']
        tab = batch['tab']

        # (B, C, H, C) -> (B, N, D)
        img = self.process_img(img)
        img_cls = torch.rand

        # (B, F, D)
        tab = [self.tab_embedder[fname](tab[fname]) for fname in tab]


        mask = np.random.rand((img.shape[1], self.config.mask_ratio)) > 0.5
        img_mask_tokens = torch.gather()



