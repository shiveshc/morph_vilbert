import torch
import torch.nn as nn

from ml_collections import ConfigDict


class MLPDecoder(nn.Module):
    def __init__(self,
                 config:ConfigDict) -> None:
        super().__init__()

        self.config = config

        layers = []
        num_layers = config.model.decoder.mlp_num_inner_layers
        layers.append(nn.Linear(config.model.encoder.model_dim, config.model.decoder.mlp_inner_dim))
        layers.append(nn.ReLU())
        for n in range(num_layers - 1):
            layers.append(nn.Linear(config.model.decoder.mlp_inner_dim, config.model.decoder.mlp_inner_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config.model.decoder.mlp_inner_dim,
                                config.data.in_channels*config.data.patch_size**2))

        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

        

class ConvDecoder(nn.Module):
    def __init__(self,
                 config:ConfigDict) -> None:
        super().__init__()

        self.config = config
        self.num_upsample_layers = config.data.img_size//config.data.patch_size

        layers = []
        for n in self.num_upsample_layers:
            in_ch = config.model.model_dim
            out_ch = config.model.model_dim//2
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, ))


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x