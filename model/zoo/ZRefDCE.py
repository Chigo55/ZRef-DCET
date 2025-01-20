import torch
import torch.nn as nn
import torch.nn.functional as F

from block import ConvolutionBlock, ConvolutionConcatBlock


class ZRDCE(nn.Module):

    def __init__(self, in_channels=3, out_channels=24, hidden_channels=32):
        super(ZRDCE, self).__init__()

        self.conv_input = ConvolutionBlock(in_channels=in_channels, out_channels=hidden_channels)
        self.conv_block = ConvolutionBlock(in_channels=hidden_channels, out_channels=hidden_channels)
        self.conv_concat_block = ConvolutionConcatBlock(in_channels=hidden_channels * 2, out_channels=hidden_channels)
        self.conv_concat_last_block = ConvolutionConcatBlock(in_channels=hidden_channels * 2, out_channels=out_channels)

    def forward(self, x: torch.Tensor):
        x1 = self.conv_input(x)
        x2 = self.conv_block(x1)
        x3 = self.conv_block(x2)
        x4 = self.conv_block(x3)
        x5 = self.conv_concat_block(x3, x4)
        x6 = self.conv_concat_block(x2, x5)
        x7 = self.conv_concat_last_block(x1, x6)

        x_r = F.tanh(input=x7)
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(tensor=x_r, split_size_or_sections=3, dim=1)

        x = x + r1 * (torch.pow(input=x, exponent=2) - x)
        x = x + r2 * (torch.pow(input=x, exponent=2) - x)
        x = x + r3 * (torch.pow(input=x, exponent=2) - x)
        x = x + r4 * (torch.pow(input=x, exponent=2) - x)
        x = x + r5 * (torch.pow(input=x, exponent=2) - x)
        x = x + r6 * (torch.pow(input=x, exponent=2) - x)
        x = x + r7 * (torch.pow(input=x, exponent=2) - x)
        enhance_image = x + r8 * (torch.pow(input=x, exponent=2) - x)
        r = torch.cat(tensors=[r1, r2, r3, r4, r5, r6, r7, r8], dim=1)
        return enhance_image, r
