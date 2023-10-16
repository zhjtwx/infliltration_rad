import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self.drop_prob / (self.block_size ** 2)
            sh = x.shape[2]
            gamma *= (sh / (sh - self.block_size + 1))**2
            
            # sample mask
            mask = (torch.rand(*x.shape[0:]).cuda(x.device) < gamma).float()

            # place mask on input device
           
            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, :, :, :]

            # scale output
            normalize_factor = block_mask.numel()/block_mask.shape[0]/(block_mask.sum(-1).sum(-1).sum(-1))
            normalize_factor= normalize_factor.view((normalize_factor.shape[0],1,1,1))
            out = out * normalize_factor

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask,
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask

        return block_mask

class DropBlock3D(DropBlock2D):
    r"""Randomly zeroes 3D spatial blocks of the input tensor.

    An extension to the concept described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, D, H, W)`
        - Output: `(N, C, D, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock3D, self).__init__(drop_prob, block_size)

    def forward(self, x):
        # shape: (bsize, channels, depth, height, width)

        assert x.dim() == 5, \
            "Expected input with 5 dimensions (bsize, channels, depth, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self.drop_prob / (self.block_size ** 2)
            sh = x.shape[2]
            gamma *= (sh / (sh - self.block_size + 1))**3
            # sample mask
            mask = (torch.rand(*x.shape[0:]).cuda(x.device) < gamma).float()

            # place mask on input device
            

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, :, :, :, :]

            # scale output
            normalize_factor = block_mask.numel()/block_mask.shape[0]/(block_mask.sum(-1).sum(-1).sum(-1).sum(-1))
            normalize_factor= normalize_factor.view((normalize_factor.shape[0],1,1,1,1))
            out = out * normalize_factor

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool3d(input=mask,
                                  kernel_size=(self.block_size, self.block_size, self.block_size),
                                  stride=(1, 1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1, :-1]

        block_mask = 1 - block_mask

        return block_mask


class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        self.i = self.i + 1
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]
        