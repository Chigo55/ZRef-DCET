class Metrics:
    def __init__(self):
        super(Metrics, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def CalcAveragePSNR(self, original_image: torch.Tensor, enhanced_image: torch.Tensor):
        mse = torch.mean((original_image - enhanced_image) ** 2)

        max_pixel_value = 255.0

        psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
        return psnr

    def CalcLPIPS(self, model: torch.nn.Module, original_image: torch.Tensor, enhanced_image: torch.Tensor):
        import lpips

        loss_fn = lpips.LPIPS(net=model).to(device=self.device)
        original_image = original_image.unsqueeze(dim=0).to(device=self.device)
        enhanced_image = enhanced_image.unsqueeze(dim=0).to(device=self.device)
        lpips_distance = loss_fn(original_image, enhanced_image)
        return lpips_distance

    def CalcSSIM(self, original_image: torch.Tensor, enhanced_image: torch.Tensor):
        original_image = original_image.float()
        enhanced_image = enhanced_image.float()
        K1 = 0.01
        K2 = 0.03
        L = 1
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        mu1 = torch.mean(original_image)
        mu2 = torch.mean(enhanced_image)
        sigma1 = torch.var(original_image)
        sigma2 = torch.var(enhanced_image)
        sigma12 = torch.mean((original_image - mu1) * (enhanced_image - mu2))
        ssim_value = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
        return ssim_value

    def CalcNIQE(self, enhanced_image: torch.Tensor, patch_size: int = 8):
        enhanced_image = enhanced_image.float()
        mu = torch.nn.functional.avg_pool2d(enhanced_image, kernel_size=patch_size, stride=patch_size)
        sigma = torch.nn.functional.avg_pool2d((enhanced_image - mu) ** 2, kernel_size=patch_size, stride=patch_size)
        niqe_score = torch.mean(torch.sqrt(sigma + 1e-10))
        return niqe_score

    def CalcBRISQUE(self, enhanced_image: torch.Tensor):
        pass
