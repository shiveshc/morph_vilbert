import torch
import matplotlib.pyplot as plt


class PatchImg():
    def __init__(self,
                 img_size:int,
                 patch_size:int) -> None:
        self.img_size = img_size
        self.patch_size = patch_size
        
    def patchify(self,
                 x:torch.Tensor) -> torch.Tensor:
        '''
        patchifies image
        x - (B, C, H, W)
        '''
        assert self.img_size == x.shape[2]
        assert self.img_size == x.shape[3]
        nx = self.img_size//self.patch_size
        ny = self.img_size//self.patch_size
        x = torch.reshape(x, (x.shape[0], x.shape[1], nx, self.patch_size, ny, self.patch_size))
        x = torch.permute(x, (0, 1, 2, 4, 3, 5))
        x = torch.reshape(x, (x.shape[0], x.shape[1], nx, ny, -1))
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1, x.shape[4]))
        return x
    
    
    def unpatchify(self,
                   x:torch.Tensor) -> torch.Tensor:
        '''
        unpatchifies image
        x - (B, C, nx*ny, ps*ps)
        '''
        nx = self.img_size//self.patch_size
        ny = self.img_size//self.patch_size
        assert x.shape[2] == nx*ny
        assert x.shape[3] == self.patch_size**2
        
        x = torch.reshape(x, (x.shape[0], x.shape[1], nx, ny, self.patch_size, self.patch_size))
        x = torch.permute(x, (0, 1, 2, 4, 3, 5))
        x = torch.reshape(x, (x.shape[0], x.shape[1], nx*self.patch_size, ny*self.patch_size))
        x.shape
    
    
    def plot_patches(self,
                     x:torch.tensor) -> None:
        '''
        plot patches of patchified image
        x - (B, C, nx*ny, ps*ps)
        '''
        tmp = torch.reshape(x, (x.shape[0], x.shape[1], -1, self.patch_size, self.patch_size))
        fig, ax = plt.subplots(4, 4, figsize=(4, 4))
        for i in range(x.shape[2]):
            s_row = i//4
            s_col = i%4
            ax[s_row, s_col].imshow(tmp[0, 0, i])
            ax[s_row, s_col].axis('off')
        fig.tight_layout()