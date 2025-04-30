import numpy as np
import torch
import torch.nn as nn
from ml_collections import ConfigDict

from morph_mm.models.transformer import Transformer
from morph_mm.models.decoder import MLPDecoder
from morph_mm.utils import PatchImg

from typing import Dict, Tuple


class ImgBERT(nn.Module):
    def __init__(self, 
                 config:ConfigDict) -> None:
        super().__init__()

        self.config = config

        self.conv = nn.Conv2d(in_channels=config.data.in_channels,
                              out_channels=config.model.model_dim,
                              kernel_size=config.data.patch_size,
                              stride=config.data.patch_size)

        patcher = PatchImg(config.data.img_size, config.data.patch_size)
        self.encoder = Transformer(config)
        self.mask_token = nn.Parameter(torch.randn(1, config.model.model_dim))
        self.decoder = MLPDecoder

    
    def preprocess_img(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        x = torch.permute(x, (0, 2, 1))
        return x
    
    
    def forward(self, batch:Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        img = batch['img']
        img = self.preprocess_img(img)
        

        mask_param = torch.tensor(np.random.rand(img.shape[1]))
        idx = torch.argsort(mask_param)
        restore_idx = torch.argsort(idx)
        num_mask_tokens = int(self.config.model.mask_ratio*img.shape[1])


        img = img[:, idx, :]
        mask_tokens = self.mask_token[None, :, :].expand(-1, num_mask_tokens, -1)
        take = img[:, 0:num_mask_tokens, :]
        keep = img[:, num_mask_tokens::, :]
        img = torch.concatenate([mask_tokens, keep], dim=1)
        img = img[:, restore_idx, :]
        
        img = self.encoder(img)
        
        pred = img[:, idx, :][:, 0:num_mask_tokens, :]

        return pred, take

        


