import numpy as np
import torch
import torch.nn as nn
from ml_collections import ConfigDict

from morph_mm.models.transformer import Transformer
from morph_mm.models.decoder import MLPDecoder
from morph_mm.utils import PatchImg, PosEmbed

from typing import Dict, Tuple


class ImgBERT(nn.Module):
    def __init__(self, config:ConfigDict) -> None:
        super().__init__()

        self.config = config

        self.conv = nn.Conv2d(in_channels=config.data.in_channels,
                              out_channels=config.model.encoder.model_dim,
                              kernel_size=config.data.patch_size,
                              stride=config.data.patch_size)
        
        self.patcher = PatchImg(config.data.img_size, config.data.patch_size, config.data.in_channels)
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.model.encoder.model_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.model.encoder.model_dim))

        self.num_tokens = (config.data.img_size//config.data.patch_size)**2
        self.num_mask_tokens = int(self.num_tokens*config.model.mask_ratio)
        if config.model.encoder.pos_embed == 'sine':
            self.pos_embed = PosEmbed.sin_embed(config.model.encoder.model_dim, num_tokens)
        elif config.model.encoder.pos_embed == 'learned':
            self.pos_embed = PosEmbed.learned_embed(config.model.encoder.model_dim, num_tokens)

        self.proj = nn.Linear(config.data.in_channels*config.data.patch_size**2, config.model.encoder.model_dim)

        self.encoder = Transformer(config)
        self.decoder = MLPDecoder(config)

    
    def preprocess_img(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        x = torch.permute(x, (0, 2, 1))
        return x
    
    
    def forward(self, batch:Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = batch['img']
        print('1', img.shape)
        img = self.patcher.patchify(img)
        print('2', img.shape)
        

        mask_param = torch.tensor(np.random.rand(img.shape[0], img.shape[1]))
        idx = torch.argsort(mask_param, dim=1)
        restore_idx = torch.argsort(idx, dim=1)
        num_mask_tokens = self.num_mask_tokens

        img = torch.gather(img, dim=1, index=idx[:, :, None].repeat(1, 1, img.shape[2]))
        print('3', img.shape)
        gt = img[:, 0:num_mask_tokens, :]
        print('4', gt.shape)
        img = img[:, num_mask_tokens::, :]
        print('5', img.shape)
        
        img = self.proj(img)
        print('6', img.shape)
        mask_tokens = self.mask_token.repeat(img.shape[0], num_mask_tokens, 1)
        img = torch.concat([mask_tokens, img], dim=1)
        print('7', img.shape)
        img = torch.gather(img, dim=1, index=restore_idx[:, :, None].repeat(1, 1, img.shape[2]))
        print('8', img.shape)
        img = torch.concat([self.cls_token.repeat(img.shape[0], 1, 1), img], dim=1)
        print('9', img.shape)
        img = self.encoder(img)
        print('10', img.shape)
        
        cls_token = img[:, 0, :]
        print('11', cls_token.shape)
        img = img[:, 1::, :]
        print('12', img.shape)
        pred = torch.gather(img, dim=1, index=idx[:, :, None].repeat(1, 1, img.shape[2]))
        print('13', pred.shape)
        pred = pred[:, 0:num_mask_tokens, :]
        print('14', pred.shape)
        pred = self.decoder(pred)
        print('15', pred.shape)

        print('pred', pred.shape)
        print('gt', gt.shape)
        print('cls_token', cls_token.shape)

        return pred, gt, cls_token

        


