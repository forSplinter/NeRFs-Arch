import torch 
import torch.nn as nn
from typing import Tuple, Union, Literal, Optional, List 
from pathlib impo

class Metrics:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.lpips_alex = lpips.LPIPS(net='alex').to(device)
        self.lpips_vgg = lpips.LPIPS(net='vgg').to(device)
    
    @staticmethod
    def mse(pred: torch.Tensor, target: torch.Tensor,reduction: Literal['mean', 'sum', 'none'] = 'mean') -> torch.Tensor:
        """_summary_

        Args:
            pred (torch.Tensor): _description_
            target (torch.Tensor): _description_
            reduction (Literal[&#39;mean&#39;, &#39;sum&#39;, &#39;none&#39;], optional): _description_. Defaults to 'mean'.

        Returns:
            torch.Tensor: _description_
        """
        diff = torch.mean((pred - target) ** 2, dim=-1)
        if reduction == 'mean':
            return torch.mean(diff)
        elif reduction == 'sum':
            return torch.sum(diff)
        else:
            return diff
    
    @staticmethod
    def psnr(mse: Union[torch.Tensor, float]) -> torch.Tensor:
        """_summary_

        Args:
            mse (Union[torch.Tensor, float]): _description_

        Returns:
            torch.Tensor: _description_
        """
        if isinstance(mse, float):
            mse = torch.tensor(mse)

        return -10.0 * torch.log10(mse)
    
    def _prepare_images(img1: torch.Tensor, img2: torch.Tensor, format: Literal['HWC','NCHW', 'NHWC'] = 'NCHW')-> tuple[torch.Tensor, torch.Tensor]:
        if format == 'HWC':
            img1 = img1.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            img2 = img2.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        elif format == 'NHWC':
            img1 = img1.permute(0, 3, 1, 2)  # (N, 3, H, W)
            img2 = img2.permute(0, 3, 1, 2)  # (N, 3, H, W)
            
        return img1, img2
    
    def ssim(self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11,
             size_average: bool = True, format: Literal['HWC', 'NCHW', 'NHWC'] = 'NCHW') -> torch.Tensor:
        """_summary_

        Args:
            pred (torch.Tensor): _description_
            target (torch.Tensor): _description_
            window_size (int, optional): _description_. Defaults to 11.
            size_average (bool, optional): _description_. Defaults to True.
            format (Literal[&#39;HWC&#39;, &#39;NCHW&#39;, &#39;NHWC&#39;], optional): _description_. Defaults to 'NCHW'.

        Returns:
            torch.Tensor: _description_
        """
        pred, target = self.__prepare__images(pred, target, format)
        C1 = 0.01.pow(2)
        C2 = 0.03.pow(2)
        
        mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
        mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(pred.pow(2), window_size, stride=1, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(target.pow(2), window_size, stride=1, padding=window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean() if size_average else ssim_map.mean(dim=(1, 2, 3))
    
    
    def lpips(self, pred: torch.Tensor, target:torch.Tensor, net: Literal['alex', 'vgg'] = 'alex',
              format: Literal['HWC', 'NCHW', 'NHWC'] = 'NCHW') -> torch.Tensor:
        """_summary_

        Args:
            pred (torch.Tensor): _description_
            target (torch.Tensor): _description_
            net (Literal[&#39;alex&#39;, &#39;vgg&#39;], optional): _description_. Defaults to 'alex'.
            format (Literal[&#39;HWC&#39;, &#39;NCHW&#39;, &#39;NHWC&#39;], optional): _description_. Defaults to 'NCHW'.

        Returns:
            torch.Tensor: _description_
        """
        pred, target = self.__prepare_images(pred, target, format)
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        pred = pred * 2.0 - 1.0

        if net == 'alex':
            return self.lpips_alex(pred, target)
        else:
            return self.lpips_vgg(pred, target)
        
    def compute_all(self, pred: torch.Tensor, target: torch.Tensor,
                    format: Literal['HWC', 'NCHW', 'NHWC'] = 'NCHW') -> dict[str, torch.Tensor]:
        """_summary_

        Args:
            pred (torch.Tensor): _description_
            target (torch.Tensor): _description_
            format (Literal[&#39;HWC&#39;, &#39;NCHW&#39;, &#39;NHWC&#39;], optional): _description_. Defaults to 'NCHW'.

        Returns:
            dict[str, torch.Tensor]: _description_
        """

        with torch.no_grad():
            mse_val = self.mse(pred, target)
            psnr_val = self.psnr(mse_val)
            ssim_val = self.ssim(pred, target, format=format)
            lpips_val = self.lpips(pred, target, format=format)
        
        return {
            'mse': mse_val,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'lpips': lpips_val
        }

def img2mse(pred: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """_summary_

    Args:
        pred (torch.Tensor): _description_
        target (torch.Tensor): _description_
        reduction (str, optional): _description_. Defaults to 'mean'.

    Returns:
        torch.Tensor: _description_
    """
    return Metrics.mse(pred, target, reduction=reduction)

def mse2psnr(mse: Union[torch.Tensor, float]) -> torch.Tensor:
    """_summary_

    Args:
        mse (Union[torch.Tensor, float]): _description_

    Returns:
        torch.Tensor: _description_
    """
    return Metrics.psnr(mse)
