import torch
import torch.nn as nn

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