# train.py

import time
import torch
import torch.optim as optim
from pathlib import Path
from typing import Optional, Literal

from models.mipnerf import MipNeRF
from utils.dataset import RayNeRFDataset
from utils.metrics import img2mse, mse2psnr
from utils.export import to8b
from utils.evaluation import NeRFEvaluator


def train_one_step(batch: dict, model: torch.nn.Module, optimizer: torch.optim.Optimizer, near: float, far: float, radii: float)-> dict:

    model.train()
    optimizer.zero_grad()
    ret_dict = model(batch['rays'], (near, far), radii=radii)
    loss = img2mse(ret_dict['rgb'], batch['target_s'])
    if 'rgb0' in ret_dict:
        loss = loss + img2mse(ret_dict['rgb0'], batch['target_s'])
    loss.backward()
    optimizer.step()
    
    return {'loss': loss.item(), 'psnr': mse2psnr(loss).item()}


@torch.no_grad()
def eval_one_view(model: torch.nn.Module, dataset: RayNeRFDataset, idx: int, near: float, far: float, radii: float, device: str):
    model.eval()
    data = dataset[idx]
    rays = data['rays']
    H, W = dataset.height_width()
    
    # Extract rays
    rays_o, rays_d = (rays[0], rays[1]) if rays.shape[0] == 2 else (rays[:, 0], rays[:, 1])
    rays_o, rays_d = rays_o.to(device), rays_d.to(device)
    
    # Render
    bounds = torch.tensor([[near, far]], device=device).expand(rays_o.shape[0], 2)
    radii_tensor = torch.full((rays_o.shape[0],), radii, device=device)
    outputs = model(rays_o, rays_d, bounds, radii_tensor)
    
    rgb = outputs.get('rgb', outputs.get('rgb0')).reshape(H, W, 3).cpu()
    return rgb


def save_checkpoint(path: str, step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """Save checkpoint"""
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: str) -> int:
    """Load checkpoint"""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('step', 0)


def train(
    model_type: Literal['mipnerf'] = 'mipnerf',
    data_path: str = 'data/transforms_train.json',
    test_data_path: str = 'data/transforms_test.json',
    run_dir: str = 'runs/exp1',
    batch_size: int = 1024,
    max_steps: int = 200000,
    lr: float = 5e-4,
    device: str = 'cuda',
    i_print: int = 100,
    i_weights: int = 10000,
    i_testset: int = 50000,
    resume_from: Optional[str] = None,
    **model_kwargs
):
    """Main training loop"""
    
    # Setup
    run_dir = Path(run_dir)
    (run_dir / 'checkpoints').mkdir(exist_ok=True, parents=True)
    
    train_dataset = RayNeRFDataset(json_path=data_path, split='train', device=device)
    test_dataset = RayNeRFDataset(json_path=test_data_path, split='test', device=device)
    near, far = train_dataset.near_far()
    radii = train_dataset.radii()
    
    model = MipNeRF(**model_kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    step = load_checkpoint(resume_from, model, optimizer, device) if resume_from else 0
    
    print(f"Training from step {step} to {max_steps}")
    
    time0 = time.time()
    while step < max_steps:
        step += 1
        batch = train_dataset.get_full_batch(batch_size)
        metrics = train_one_step(batch, model, optimizer, near, far, radii)
        
        if step % i_print == 0:
            dt = time.time() - time0
            time0 = time.time()
            print(f"[{step}/{max_steps}] Loss: {metrics['loss']:.4f} "
                  f"PSNR: {metrics['psnr']:.2f} Time: {dt/i_print:.3f}s")
        
        # Checkpoint
        if step % i_weights == 0:
            path = run_dir / 'checkpoints' / f'{step:08d}.ckpt'
            save_checkpoint(path, step, model, optimizer)
            print(f"Saved: {path}")
        
        # Evaluate
        if step % i_testset == 0:
            print("Evaluating...")
            evaluator = NeRFEvaluator(model, test_dataset, device, run_dir / f'eval_{step:08d}')
            results = evaluator.evaluate_dataset(save_images=True, verbose=False)
            print(f"Test PSNR: {results['psnr']['mean']:.2f}")
    
    save_checkpoint(run_dir / 'checkpoints' / 'final.ckpt', step, model, optimizer)
    print("Training complete!")