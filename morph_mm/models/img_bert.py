import numpy as np
import torch
import torch.nn as nn
from ml_collections import ConfigDict

from morph_mm.models.transformer import Transformer
from morph_mm.models.decoder import MLPDecoder
from morph_mm.utils import PatchImg, PosEmbed, SinCos2DPosEmbed, Learned2DPosEmbed

from typing import Dict, Tuple


class ImgBERT(nn.Module):
    """
    Vision Transformer (BERT-style) model for masked image modeling.

    Args:
        config (ConfigDict): Configuration with data/model parameters.
    """
    def __init__(self, config:ConfigDict) -> None:
        super().__init__()

        self.config = config

        self.patcher = PatchImg(config.data.img_size, config.data.patch_size, config.data.in_channels)
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.model.encoder.model_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.model.encoder.model_dim))

        h = w = config.data.img_size//config.data.patch_size
        num_tokens = h*w
        self.num_mask_tokens = int(num_tokens*config.model.mask_ratio)
        if config.model.encoder.pos_embed == 'sine_cos':
            self.pos_embed = SinCos2DPosEmbed(config.model.encoder.model_dim, h, w)
        elif config.model.encoder.pos_embed == 'learned':
            self.pos_embed = Learned2DPosEmbed(config.model.encoder.model_dim, h, w)
        else:
            raise NotImplementedError(f'{config.model.encoder.pos_embed} is not implemented')

        self.proj = nn.Linear(config.data.in_channels*config.data.patch_size**2, config.model.encoder.model_dim)

        self.encoder = Transformer(config)
        self.decoder = MLPDecoder(config)
 
    
    def forward(self,
                batch:Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_img = batch['img']
        img = self.patcher.patchify(orig_img)
        

        # select randomly mask tokens
        mask_param = torch.tensor(np.random.rand(img.shape[0], img.shape[1]))
        idx = torch.argsort(mask_param, dim=1)
        restore_idx = torch.argsort(idx, dim=1)
        num_mask_tokens = self.num_mask_tokens

        # gather non mask tokens
        img = torch.gather(img, dim=1, index=idx[:, :, None].repeat(1, 1, img.shape[2]))
        gt = img[:, 0:num_mask_tokens, :]
        keep_img = img[:, num_mask_tokens::, :]
        
        # add mask tokens and encode
        x = self.proj(keep_img)
        mask_tokens = self.mask_token.repeat(x.shape[0], num_mask_tokens, 1)
        x = torch.concat([mask_tokens, x], dim=1)
        x = torch.gather(x, dim=1, index=restore_idx[:, :, None].repeat(1, 1, x.shape[2]))
        x = x + self.pos_embed()
        x = torch.concat([self.cls_token.repeat(x.shape[0], 1, 1), x], dim=1)
        x = self.encoder(x)
        
        # decode mask tokens
        cls_token = x[:, 0, :]
        x = x[:, 1::, :]
        x = torch.gather(x, dim=1, index=idx[:, :, None].repeat(1, 1, x.shape[2]))
        pred = x[:, 0:num_mask_tokens, :]
        pred = self.decoder(pred)

        # reconstruct image
        recon_img = torch.concat([pred, keep_img], dim=1)
        recon_img = torch.gather(recon_img, dim=1, index=restore_idx[:, :, None].repeat(1, 1, recon_img.shape[2]))
        recon_img = self.patcher.unpatchify(recon_img)

        return pred, gt, cls_token, orig_img, recon_img

        


