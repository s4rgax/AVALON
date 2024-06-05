"""
File contenente la definizione della classe CNN, la quale rappresenta l'intera rete
convoluzionale.
"""
import torch
import torch.nn as nn

from src.cnn.custom_layers import PixelAttention2D, EfficientChannelAttention2D


class CNN(nn.Module):
    """
    Metodo costruttore della classe.

    :param hyperParams: iperparametri usati durante la definizione della CNN (Vengono ottimizzati).
    :param params: parametri usati durante la definizione della CNN (Restano fissi).
    """
    def __init__(self, hyperParams: dict, params: dict, size: int) -> None:
        super(CNN, self).__init__()

        self.num_conv = int(hyperParams["num_conv"])
        self.convs = nn.ModuleList()

        in_channels = int(params.get('nChannels'))
        out_channels = 32

        if int(params.get('setChannelAttentionLayer')) == 1:
            self.set_channel_attention_layer = int(params.get('setChannelAttentionLayer')) == 1
            self.channel_attention = EfficientChannelAttention2D(nf=in_channels)

        for n in range(self.num_conv):
            self.convs.append(nn.Conv2d(in_channels = in_channels,
                               out_channels = out_channels,
                               kernel_size = int(hyperParams["kernel"]),
                               padding = 'same'))
            in_channels = out_channels
            out_channels = in_channels*2

        if int(params.get('setAttentionLayer')) == 1:
            self.set_attention_layer = int(params.get('setAttentionLayer')) == 1
            self.pixel_attention = PixelAttention2D(in_channels, size)
            num_linear_in_channels = int(size) * int(size)
        else:
            num_linear_in_channels = in_channels * int(size) * int(size)

        self.fc0 = nn.Linear(in_features = num_linear_in_channels,
                             out_features = 256)
        self.fc1 = nn.Linear(in_features = 256,
                             out_features = 1024)
        self.fc2 = nn.Linear(in_features = 1024, out_features = int(params["nClasses"]))

        # PER SIGMOID
        # self.fc2 = nn.Linear(in_features=1024, out_features=1)
        # self.fl = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(hyperParams["dropout"])
        self.flatten = nn.Flatten()

    def forward(self, x, get_explanation =False):
        out= x

        channel_att_map= None
        if hasattr(self,'set_channel_attention_layer')and  self.set_channel_attention_layer:
            out = self.channel_attention(out)
            channel_att_map = out

        for n in range(self.num_conv-1):
            out = self.relu(self.convs[n](out))
            out = self.dropout(out)

        out = self.convs[self.num_conv-1](out)

        pixel_att_map = None
        if hasattr(self, 'set_attention_layer') and self.set_attention_layer:
            out = self.pixel_attention(out)
            out = torch.mean(out, dim=[1])
            pixel_att_map = out

        out = self.relu(out)
        out = self.flatten(out)

        out = self.relu(self.fc0(out))
        out = self.relu(self.fc1(out))
        out = self.fc2(out)

        # out = self.fl(self.fc2(out))
        if get_explanation:
            return out, pixel_att_map, channel_att_map
        else:
            return out
