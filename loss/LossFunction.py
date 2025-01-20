import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorConstancyLoss(nn.Module):
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, image):
        mean_rgb = torch.mean(image, [2, 3], keepdim=True)  # Mean for each channel
        mean_r, mean_g, mean_b = torch.split(mean_rgb, 1, dim=1)  # Split channels
        diff_rg = torch.pow(mean_r - mean_g, 2)
        diff_rb = torch.pow(mean_r - mean_b, 2)
        diff_gb = torch.pow(mean_g - mean_b, 2)
        loss = torch.pow(torch.pow(diff_rg, 2) + torch.pow(diff_rb, 2) + torch.pow(diff_gb, 2), 0.5)  # Aggregate loss

        return loss


class SpatialConsistencyLoss(nn.Module):
    def __init__(
        self,
    ):
        super(SpatialConsistencyLoss, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Define convolutional kernels for detecting directional differences
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).to(self.device).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).to(self.device).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).to(self.device).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).to(self.device).unsqueeze(0).unsqueeze(0)

        # Register kernels as non-trainable parameters
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.avg_pool = nn.AvgPool2d(4)

    def forward(self, original_image, enhanced_image):
        original_mean = torch.mean(original_image, 1, keepdim=True)
        enhanced_mean = torch.mean(enhanced_image, 1, keepdim=True)

        # Downsample images for efficiency
        original_pooled = self.avg_pool(original_mean)
        enhanced_pooled = self.avg_pool(enhanced_mean)

        # Compute differences for all directions
        diff_left = F.conv2d(original_pooled, self.weight_left, padding=1) - F.conv2d(
            enhanced_pooled, self.weight_left, padding=1
        )
        diff_right = F.conv2d(original_pooled, self.weight_right, padding=1) - F.conv2d(
            enhanced_pooled, self.weight_right, padding=1
        )
        diff_up = F.conv2d(original_pooled, self.weight_up, padding=1) - F.conv2d(
            enhanced_pooled, self.weight_up, padding=1
        )
        diff_down = F.conv2d(original_pooled, self.weight_down, padding=1) - F.conv2d(
            enhanced_pooled, self.weight_down, padding=1
        )

        # Combine directional differences into the final loss
        loss = torch.pow(diff_left, 2) + torch.pow(diff_right, 2) + torch.pow(diff_up, 2) + torch.pow(diff_down, 2)

        return loss


class ExposureControlLoss(nn.Module):
    def __init__(self, patch_size, target_mean):
        super(ExposureControlLoss, self).__init__()
        self.avg_pool = nn.AvgPool2d(patch_size)
        self.target_mean = target_mean
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, image):
        mean_intensity = torch.mean(image, 1, keepdim=True)  # Compute mean intensity per channel
        pooled_mean = self.avg_pool(mean_intensity)  # Pool over patches
        loss = torch.mean(torch.pow(pooled_mean - torch.FloatTensor([self.target_mean]).to(self.device), 2))

        return loss


class TotalVariationLoss(nn.Module):
    def __init__(self, weight=1):
        super(TotalVariationLoss, self).__init__()
        self.weight = weight

    def forward(self, image):
        batch_size, _, height, width = image.size()
        count_h = (height - 1) * width
        count_w = height * (width - 1)

        # Compute variations in horizontal and vertical directions
        h_variation = torch.pow((image[:, :, 1:, :] - image[:, :, :-1, :]), 2).sum()
        w_variation = torch.pow((image[:, :, :, 1:] - image[:, :, :, :-1]), 2).sum()

        # Normalize by the total number of pixels
        loss = self.weight * 2 * (h_variation / count_h + w_variation / count_w) / batch_size

        return loss


class LuminanceLoss(nn.Module):
    def __init__(self):
        super(LuminanceLoss, self).__init__()

    def forward(self, original_image, enhanced_image):
        # Compute luminance for original and enhanced images
        original_luminance = (
            0.299 * original_image[:, 0, :, :] + 0.587 * original_image[:, 1, :, :] + 0.114 * original_image[:, 2, :, :]
        )
        enhanced_luminance = (
            0.299 * enhanced_image[:, 0, :, :] + 0.587 * enhanced_image[:, 1, :, :] + 0.114 * enhanced_image[:, 2, :, :]
        )

        # Compute mean squared error between luminances
        loss = torch.mean((original_luminance - enhanced_luminance) ** 2)

        return loss


class PhysicalConstraintLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(PhysicalConstraintLoss, self).__init__()
        self.weight = weight

    def forward(self, enhanced_image):
        # Penalize pixel values outside [0, 1]
        loss = self.weight * torch.mean(
            torch.clamp(enhanced_image - 1, min=0) ** 2 + torch.clamp(-enhanced_image, min=0) ** 2
        )

        return loss


class MultiscaleLoss(nn.Module):
    def __init__(self, scales=3):
        super(MultiscaleLoss, self).__init__()
        self.scales = scales
        self.avg_pool = nn.AvgPool2d(2, stride=2)

    def forward(self, original_image, enhanced_image):
        loss = 0
        for _ in range(self.scales):
            original_image = self.avg_pool(original_image)  # Downsample original image
            enhanced_image = self.avg_pool(enhanced_image)  # Downsample enhanced image
            loss += torch.mean(torch.abs(original_image - enhanced_image))  # Compute error at current scale

        return loss


class TotalLoss(nn.Module):
    def __init__(self, initial_weights=None):
        super(TotalLoss, self).__init__()

        # Initialize individual loss components
        self.tv_loss = TotalVariationLoss()
        self.spa_loss = SpatialConsistencyLoss()
        self.color_loss = ColorConstancyLoss()
        self.exp_loss = ExposureControlLoss(patch_size=16, target_mean=0.6)
        self.luminance_loss = LuminanceLoss()
        self.physical_loss = PhysicalConstraintLoss()
        self.multiscale_loss = MultiscaleLoss()

        # Initialize learnable weights for each loss
        if initial_weights is None:
            initial_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.5]
        self.raw_weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))

    def forward(self, img_lowlight, enhanced_image, A):
        # Normalize the weights using softmax
        normalized_weights = F.softmax(self.raw_weights, dim=0)

        # Calculate each individual loss component
        loss_tv = self.tv_loss(A)
        loss_spa = torch.mean(self.spa_loss(img_lowlight, enhanced_image))
        loss_color = torch.mean(self.color_loss(enhanced_image))
        loss_exp = torch.mean(self.exp_loss(enhanced_image))
        loss_luminance = torch.mean(self.luminance_loss(img_lowlight, enhanced_image))
        loss_physical = torch.mean(self.physical_loss(enhanced_image))
        loss_multiscale = torch.mean(self.multiscale_loss(img_lowlight, enhanced_image))

        # Compute the total loss as a weighted sum of all components
        total_loss = (
            normalized_weights[0] * loss_tv
            + normalized_weights[1] * loss_spa
            + normalized_weights[2] * loss_color
            + normalized_weights[3] * loss_exp
            + normalized_weights[4] * loss_luminance
            + normalized_weights[5] * loss_physical
            + normalized_weights[6] * loss_multiscale
        )

        return total_loss, normalized_weights
