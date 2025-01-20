import torch
import torch.nn as nn
import torch.nn.functional as F


class IlluminationGuidedMultiHeadSelfAttension(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super(IlluminationGuidedMultiHeadSelfAttension, self).__init__()

        self.FullyConnectedLayer = nn.Linear(in_features=in_features, out_features=hidden_features, bias=False)

    def forward(self, x_in, light_up_feature):
        bx, hx, wx, cx = x_in.shape()
        by, hy, wy, cy = light_up_feature.shape()
        x_in.reshape(bx, hx * wx, cx)
        light_up_feature.reshape(by, hy * wy, cy)

        Q = self.FullyConnectedLayer(x_in)
        K = self.FullyConnectedLayer(x_in)
        V = self.FullyConnectedLayer(x_in)

        Q = Q.tranpose(-2, -1)
        K = K.tranpose(-2, -1)
        V = V.tranpose(-2, -1)


class IlluminationGuidedAttension(nn.Module):
    def __init__(
        self,
    ):
        super(IlluminationGuidedAttension, self).__init__()

    def forward(
        self,
    ):
        pass


class IlluminationEstimator(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, hidden_channels=32):
        super(IlluminationEstimator, self).__init__()

        self.ConvLayer1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1)
        self.ConvLayer2 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=5, padding=2, groups=hidden_channels
        )
        self.ConvLayer3 = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, image: torch.Tensor):
        illumination_prior = image.mean(dim=1).unsqueeze(dim=1)
        x = torch.cat(tensors=[image, illumination_prior], dim=1)
        x = self.ConvLayer1(x)
        light_up_feature = self.ConvLayer2(x)
        light_up_map = self.ConvLayer3(light_up_feature)
        return light_up_map, light_up_feature


class CorruptionRestorer(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, hidden_channels=32):
        super(CorruptionRestorer, self).__init__()
