import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity


def SSIM(output,ground_truth):
    metric = StructuralSimilarityIndexMeasure(data_range=1.0)#.cuda()
    ssim =metric(output,ground_truth)
    return ssim.item()

def LPIPS(output,ground_truth):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')#.cuda()
    fin_lpips = lpips(output,ground_truth)
    return fin_lpips

def PSNR(output,ground_truth):
    return psnr(output, ground_truth)

def psnr(img_batch, ref_batch, batched=False, factor=1.0):
    """Standard PSNR."""
    def get_psnr(img_in, img_ref):
        mse = ((img_in - img_ref)**2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10(factor**2 / mse))
        elif not torch.isfinite(mse):
            return img_batch.new_tensor(float('nan'))
        else:
            return img_batch.new_tensor(float('inf'))

    if batched:
        psnr = get_psnr(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()

    return psnr.item()

def MSE(output,ground_truth):
    return (output.detach() - ground_truth).pow(2).mean()

def trapezoidal_rule(x, y):
    n = len(x)
    area = 0

    for i in range(n - 1):
        width = x[i + 1] - x[i]
        height_avg = (y[i] + y[i + 1]) / 2
        area += width * height_avg

    return area/x[-1]