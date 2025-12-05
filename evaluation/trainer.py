# evaluation/trainer.py (version améliorée)

import time
import torch
import torch.optim as optim
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
import mlflow
import tempfile
from torch.optim.lr_scheduler import LambdaLR

from model.mip_nerf import MipNeRF
from utils.dataset import RayNeRFDataset
from utils.metrics import img2mse, mse2psnr

def train_one_step_with_logging(batch: dict, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                               scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                               near: float, far: float, radii: float, device: str, 
                               step: int, use_mlflow: bool) -> dict:
    """Train one step with MLflow logging"""
    model.train()
    optimizer.zero_grad()
    
    rays = batch['rays']
    if rays.shape[0] == 2:
        rays_o, rays_d = rays[0].to(device), rays[1].to(device)
    else:
        rays_o, rays_d = rays[:, 0].to(device), rays[:, 1].to(device)
    
    num_rays = rays_o.shape[0]
    bounds = torch.tensor([[near, far]], device=device).expand(num_rays, 2)
    radii_tensor = torch.full((num_rays,), radii, device=device)
    
    ret_dict = model(rays_o, rays_d, bounds, radii_tensor)
    
    target = batch['target_s'].to(device)
    loss = img2mse(ret_dict['rgb'], target)
    
    if 'rgb0' in ret_dict:
        loss = loss + img2mse(ret_dict['rgb0'], target)
    
    loss.backward()
    optimizer.step()
    
    if scheduler is not None:
        scheduler.step()
    
    psnr = mse2psnr(loss).item()
    
    # Log metrics to MLflow
    if use_mlflow:
        mlflow.log_metrics({
            'train_loss': loss.item(),
            'train_psnr': psnr,
            'learning_rate': optimizer.param_groups[0]['lr']
        }, step=step)
    
    return {'loss': loss.item(), 'psnr': psnr}

def save_checkpoint(path: str, step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None)-> None:
    """_summary_

    Args:
        path (str): _description_
        step (int): _description_
        model (torch.nn.Module): _description_
        optimizer (torch.optim.Optimizer): _description_
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler], optional): _description_. Defaults to None.
    """
    checkpoint = {
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    torch.save(checkpoint, path)

def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: str, 
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> int:
    """_summary_

    Args:
        path (str): _description_
        model (torch.nn.Module): _description_
        optimizer (torch.optim.Optimizer): _description_
        device (str): _description_
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler], optional): _description_. Defaults to None.

    Returns:
        int: _description_
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint['step']

def create_scheduler(lr: float, optimizer: torch.optim.Optimizer, lr_decay_steps: int, lr_decay_rate: float,
                     lr_warmup_init: float, lr_decay_warmup_step: int, max_steps: int)-> LambdaLR:
    """_summary_

    Args:
        optimizer (torch.optim.Optimizer): _description_
        lr_decay_steps (int): _description_
        lr_decay_rate (float): _description_
        lr_warmup_init (float): _description_
        lr_decay_warmup_step (int): _description_
        max_steps (int): _description_
    """
    def lr_lambda(step):
        if step < lr_decay_warmup_step:
            warmup_factor = step / lr_decay_warmup_step
            lr_scale = lr_warmup_init + (lr - lr_warmup_init) * warmup_factor
            return lr_scale / lr
        
        else:
            decay_step = (step - lr_decay_warmup_step) // lr_decay_steps
            decay_factor = lr_decay_rate ** decay_step
            return decay_factor
    
    return LambdaLR(optimizer, lr_lambda)

def current_lr(optimizer: torch.optim.Optimizer) -> float:
    """_summary_

    Args:
        optimizer (torch.optim.Optimizer): _description_

    Returns:
        float: _description_
    """
    return optimizer.param_groups[0]['lr']   

def log_sample_image(model, dataset, idx, step, device, use_mlflow):
    if not use_mlflow:
        return
    
    try:
        model.eval()
        with torch.no_grad():
            data = dataset[idx]
            rays = data['rays']
            H, W = dataset.H, dataset.W
            
            if rays.shape[0] == 2:
                rays_o = rays[0].reshape(-1, 3).to(device)
                rays_d = rays[1].reshape(-1, 3).to(device)
            else:
                rays_o = rays[:, 0].reshape(-1, 3).to(device)
                rays_d = rays[:, 1].reshape(-1, 3).to(device)
            
            bounds = torch.tensor([[dataset.near, dataset.far]], device=device).expand(rays_o.shape[0], 2)
            radii = torch.full((rays_o.shape[0],), dataset.radii(), device=device)
            
            chunk_size = 32768 
            all_rgb = []
            
            for i in range(0, rays_o.shape[0], chunk_size):
                chunk_o = rays_o[i:i+chunk_size]
                chunk_d = rays_d[i:i+chunk_size]
                chunk_bounds = bounds[i:i+chunk_size]
                chunk_radii = radii[i:i+chunk_size]
                
                outputs = model(chunk_o, chunk_d, chunk_bounds, chunk_radii)
                rgb_chunk = outputs.get('rgb', outputs.get('rgb0'))
                all_rgb.append(rgb_chunk)
            
            rgb = torch.cat(all_rgb, dim=0)
            
            rgb = rgb.reshape(H, W, 3)
            rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(rgb_np)
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                img.save(f.name)
                if use_mlflow:
                    mlflow.log_artifact(f.name, f"training_samples/step_{step}")
                    
        print(f" Preview logged (step {step})")
        
    except Exception as e:
        print(f"Failed to log sample image: {e}")


def train(
    model_type: str = 'mipnerf',
    data_path: str = 'data/transforms_train.json',
    test_data_path: str = 'data/transforms_test.json',
    run_dir = None,
    batch_size: int = 1024,
    max_steps: int = 200000,
    lr: float = 5e-4,
    device: str = 'cuda',
    i_print: int = 100,
    i_weights: int = 10000,
    i_testset: int = 50000,
    log_img_idx: int = 0,
    resume_from: Optional[str] = None,
    use_tensorboard: bool = True,
    use_mlflow: bool = False,
    lr_decay_steps: int = 50000,
    lr_decay_rate: float = 0.1,
    lr_decay_warmup_step: int = 2000,
    lr_warmup_init: float = 1e-5,
    **model_kwargs
):
    if isinstance(lr, str):
        lr = float(lr)
    
    mipnerf_kwargs = {}
    valid_keys = [
        'net_depth', 'net_width', 'net_depth_fine', 'net_width_fine',
        'N_samples', 'N_importance', 'use_viewdirs', 'use_embed',
        'multires', 'multires_views', 'ray_chunk', 'pts_chunk',
        'perturb', 'raw_noise_std', 'white_bkgd', 'use_hierarchical'
    ]
    
    for key in valid_keys:
        if key in model_kwargs:
            mipnerf_kwargs[key] = model_kwargs[key]
    
    run_dir = Path(run_dir) if run_dir else Path('runs/exp1')
    (run_dir / 'checkpoints').mkdir(exist_ok=True, parents=True)
    
    train_dataset = RayNeRFDataset(json_path=data_path, split='train', device=device)
    test_dataset = RayNeRFDataset(json_path=test_data_path, split='val', device=device)
    
    near, far = train_dataset.near, train_dataset.far
    radii = train_dataset.radii()
    
    model = MipNeRF(**mipnerf_kwargs)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = create_scheduler(
        optimizer=optimizer,
        lr=lr,
        lr_decay_steps=lr_decay_steps,
        lr_decay_rate=lr_decay_rate,
        lr_decay_warmup_step=lr_decay_warmup_step,
        lr_warmup_init=lr_warmup_init,
        max_steps=max_steps
    )
    
    step = 0
    if resume_from and Path(resume_from).exists():
        step = load_checkpoint(resume_from, model, optimizer, device)
    
    print(f"Training from step {step} to {max_steps}")
    print(f"Learning rate: {lr}")
    print(f"MLflow logging: {use_mlflow}")
    
    time0 = time.time()
    metrics = {'loss': 0.0, 'psnr': 0.0}  # Initialize metrics
    
    # Log initial sample
    if use_mlflow and step> 0:
        log_sample_image(model, test_dataset, log_img_idx, step, device, use_mlflow)
    
    while step < max_steps:
        step += 1
        
        # Get batch
        batch = train_dataset.get_full_batch(batch_size)
        
        # Train step
        metrics = train_one_step_with_logging(
            batch, model, optimizer, scheduler, near, far, radii, device, step, use_mlflow
        )
        
        # Print progress
        if step % i_print == 0:
            dt = time.time() - time0
            time0 = time.time()
            print(f"[{step}/{max_steps}] Loss: {metrics['loss']:.4f} "
                  f"PSNR: {metrics['psnr']:.2f} Time: {dt/i_print:.3f}s")
            
            # Log additional metrics
            if use_mlflow:
                mlflow.log_metric('iter_time', dt/i_print, step=step)
        
        # Save checkpoint
        if step % i_weights == 0:
            path = run_dir / 'checkpoints' / f'{step:08d}.ckpt'
            save_checkpoint(str(path), step, model, optimizer, scheduler)
            print(f"Saved: {path}")
            
            # Log checkpoint to MLflow occasionally
            if use_mlflow:
                mlflow.log_artifact(str(path), "checkpoints")
                mlflow.log_metric('final_step', step)
                mlflow.log_metric('final_loss', metrics['loss'])
                mlflow.log_metric('final_psnr', metrics['psnr'])
        
        # Evaluate and log images
        if step % i_testset == 0:
            print(f"Evaluating at step {step}...")
            
            if use_mlflow:
                log_sample_image(model, test_dataset, log_img_idx, step, device, use_mlflow)
                model.eval()
                with torch.no_grad():
                    eval_batch = test_dataset.get_full_batch(min(100, batch_size))
                    eval_rays = eval_batch['rays']
                    
                    if eval_rays.shape[0] == 2:
                        rays_o, rays_d = eval_rays[0].to(device), eval_rays[1].to(device)
                    else:
                        rays_o, rays_d = eval_rays[:, 0].to(device), eval_rays[:, 1].to(device)
                    
                    bounds = torch.tensor([[near, far]], device=device).expand(rays_o.shape[0], 2)
                    radii_tensor = torch.full((rays_o.shape[0],), radii, device=device)
                    
                    outputs = model(rays_o, rays_d, bounds, radii_tensor)
                    rgb = outputs.get('rgb', outputs.get('rgb0'))
                    
                    if 'target_s' in eval_batch:
                        target = eval_batch['target_s'].to(device)
                        eval_loss = img2mse(rgb, target).item()
                        eval_psnr = mse2psnr(eval_loss).item()
                        
                        mlflow.log_metrics({
                            'eval_loss': eval_loss,
                            'eval_psnr': eval_psnr
                        }, step=step)
                        
                        print(f"  Eval PSNR: {eval_psnr:.2f}")
    
    final_path = run_dir / 'checkpoints' / 'final.ckpt'
    save_checkpoint(str(final_path), step, model, optimizer, scheduler)
    
    if use_mlflow:
        mlflow.log_artifact(str(final_path), "checkpoints")
        mlflow.log_metric('final_step', step)
        if use_mlflow:
            mlflow.log_artifact(str(final_path), "checkpoints")
            mlflow.log_metric('final_step', step)
            mlflow.log_metric('final_loss', metrics['loss'])
            mlflow.log_metric('final_psnr', metrics['psnr'])    
    print("Training complete!")
    print(f"Final checkpoint: {final_path}")