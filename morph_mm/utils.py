import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class PatchImg():
    def __init__(self,
                 img_size:int,
                 patch_size:int,
                 n_channels:int) -> None:
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        
    def patchify(self,
                 x:torch.Tensor) -> torch.Tensor:
        '''
        patchifies image
        x - (B, C, H, W)
        out - (B, nx*ny, nch*ps*ps)
        '''
        assert self.img_size == x.shape[2]
        assert self.img_size == x.shape[3]
        nx = self.img_size//self.patch_size
        ny = self.img_size//self.patch_size
        x = torch.reshape(x, (x.shape[0], x.shape[1], nx, self.patch_size, ny, self.patch_size))
        x = torch.permute(x, (0, 2, 4, 1, 3, 5))
        x = torch.reshape(x, (x.shape[0], nx, ny, -1))
        x = torch.reshape(x, (x.shape[0], -1, x.shape[3]))
        return x
    
    
    def unpatchify(self,
                   x:torch.Tensor) -> torch.Tensor:
        '''
        unpatchifies image
        x - (B, nx*ny, nch*ps*ps)
        out - (B, C, H, W)
        '''
        nx = self.img_size//self.patch_size
        ny = self.img_size//self.patch_size
        assert x.shape[1] == nx*ny
        assert x.shape[2] == self.n_channels*(self.patch_size**2)
        
        x = torch.reshape(x, (x.shape[0], nx, ny, self.n_channels, self.patch_size, self.patch_size))
        x = torch.permute(x, (0, 3, 1, 4, 2, 5))
        x = torch.reshape(x, (x.shape[0], x.shape[1], nx*self.patch_size, ny*self.patch_size))
        return x
    
    
    def plot_patches(self,
                     x:torch.tensor) -> None:
        '''
        plot patches of patchified image
        x - (B, nx*ny, nch*ps*ps)
        '''
        tmp = torch.reshape(x, (x.shape[0], -1, self.n_channels, self.patch_size, self.patch_size))
        fig, ax = plt.subplots(4, 4, figsize=(4, 4))
        for i in range(x.shape[1]):
            s_row = i//4
            s_col = i%4
            ax[s_row, s_col].imshow(tmp[0, i, 0])
            ax[s_row, s_col].axis('off')
        fig.tight_layout()


class Learned2DPosEmbed(nn.Module):
    def __init__(self,
                 model_dim:int,
                 height:int,
                 width:int) -> None:
        super().__init__()
        self.row_embed = nn.Embedding(height, model_dim // 2)
        self.col_embed = nn.Embedding(width, model_dim // 2)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self) -> torch.Tensor:
        # shape: (H, W, D)
        h = self.row_embed(torch.arange(self.row_embed.num_embeddings))
        w = self.col_embed(torch.arange(self.col_embed.num_embeddings))
        pos = torch.cat([
            h.unsqueeze(1).expand(-1, w.size(0), -1),  # shape (H, W, D/2)
            w.unsqueeze(0).expand(h.size(0), -1, -1)   # shape (H, W, D/2)
        ], dim=-1)
        pos = torch.reshape(pos, (-1, pos.shape[-1]))
        return pos  # shape: (H*W, D)


class SinCos2DPosEmbed(nn.Module):
    def __init__(self,
                 model_dim:int,
                 height:int,
                 width:int) -> None:
        super().__init__()
        freq = torch.tensor(np.linspace(0, 2*np.pi, model_dim//4), dtype=torch.float)
        row_sin_embed = torch.stack([torch.sin(x*freq) for x in range(height)], dim=0)
        row_cos_embed = torch.stack([torch.cos(x*freq) for x in range(height)], dim=0)
        col_sin_embed = torch.stack([torch.sin(x*freq) for x in range(width)], dim=0)
        col_cos_embed = torch.stack([torch.cos(x*freq) for x in range(width)], dim=0)
        self.row_embed = torch.cat([row_sin_embed, row_cos_embed], dim=-1)
        self.col_embed = torch.cat([col_sin_embed, col_cos_embed], dim=-1)
    
    def forward(self) -> torch.Tensor:
        h = self.row_embed
        w = self.col_embed
        pos = torch.cat([
            h.unsqueeze(1).expand(-1, w.size(0), -1),  # shape (H, W, D)
            w.unsqueeze(0).expand(h.size(0), -1, -1)   # shape (H, W, D)
        ], dim=-1)
        pos = torch.reshape(pos, (-1, pos.shape[-1]))
        return pos  # shape: (H*W, D)


class PosEmbed:
    def __init__(self):
        pass

    @staticmethod
    def learned_embed(model_dim:int,
                      height:int,
                      width:int) -> torch.Tensor:
        emb = Learned2DPosEmbed(model_dim, height, width)
        return emb()
    
    
    @staticmethod
    def sin_embed(model_dim:int,
                  height:int,
                  width:int) -> torch.Tensor:
        emb = SinCos2DPosEmbed(model_dim, height, width)
    

        