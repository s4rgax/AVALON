import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelAttention2D(nn.Module):
    """Implements Pixel Attention for convolutional networks in PyTorch.
    Inputs need to be Conv2D feature maps.
    The layer implements the following:
    1. Conv2D with k=1 for fully connected features
    2. Sigmoid activation to create attention maps
    3. Element-wise multiplication to create attention activated outputs
    4. Conv2D with k=1 for fully connected features

    Args:
    * nf [int]: number of filters or channels
    * name : Name of layer
    Call Arguments:
    * Feature maps : Conv2D feature maps of the shape `[batch,C,H,W]`.
    Output:
    Attention activated Conv2D features of shape `[batch,C,H,W]`.
    """

    def __init__(self, nf, size):
        super(PixelAttention2D, self).__init__()
        self.nf = nf
        self.conv1 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=1)

    def forward(self, x):
        y = self.conv1(x)
        sig = torch.sigmoid(y)
        out = x * sig
        out = self.conv1(out)
        return out


class EfficientChannelAttention2D(nn.Module):
    """
    Implements Efficient Channel Attention (ECA) for convolutional networks in PyTorch.
    Inputs need to be Conv2D feature maps.
    The layer implements the following:
        1. Adaptive Average Pooling to create `[1,1,C]` vectors
        2. Conv1D with cross activation
        3. Sigmoid activation to create attention maps

    Args:
        nf (int): Number of filters or channels

    Call Arguments:
        Feature maps: Conv2D feature maps of the shape `[batch, C, H, W]`.

    Output:
        Attention activated Conv2D features of shape `[batch, C, H, W]`.
    """

    def __init__(self, nf):
        super(EfficientChannelAttention2D, self).__init__()
        self.nf = nf
        self.conv1 = nn.Conv1d(in_channels=nf, out_channels=nf, kernel_size=3, padding='same', bias=False)

    def forward(self, x):
        # Adaptive average pooling to create `[1,1,C]` vectors
        pool = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        pool = pool.unsqueeze(-1)
        # pool = pool.unsqueeze(0)

        # Conv1D with cross activation
        att = self.conv1(pool)

        #KERAS perm=[W,C,H]
        #KERAS perm=[0,2,1]

        #PYCAZZ perm=[]
        # Transpose and sigmoid activation to create attention maps
        # att = att.transpose(1,2)
        att = att.unsqueeze(2)
        att = torch.sigmoid(att)

        # Multiply the input by the attention maps
        y = x * att
        return y

    def get_config(self):
        config = super(EfficientChannelAttention2D, self).get_config()
        config.update({'nf': self.nf})
        return config
