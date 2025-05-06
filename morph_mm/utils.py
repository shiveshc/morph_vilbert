import torch
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


class PosEmbed:
    def __init__(self):
        pass

    @staticmethod
    def learned_embed(model_dim:int,
                      num_tokens:int) -> torch.Tensor:
        pass

    
    @staticmethod
    def sin_embed(model_dim:int,
                  num_tokens:int) -> torch.Tensor:
        pass
    

        