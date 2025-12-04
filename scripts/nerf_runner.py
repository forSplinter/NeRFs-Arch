from pathlib import Path
import argparse
import yaml
import torch
import mlflow
import shutil
from datetime import datetime
from evaluation.trainer import train
from evaluation.evaluate import Eval
from model.mip_nerf import MipNeRF
from utils.dataset import RayNeRFDataset

def parse_args():
    parser = argparse.ArgumentParser(description="NeRF Training with MLflow")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--mlflow', action='store_true', help='Enable MLflow tracking')
    parser.add_argument('--eval', action='store_true', help='Run evaluation only')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--debug', action='store_true', help='Debug mode (fewer steps)')
    return parser.parse_args()

def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def flatten_config_for_logging(config, prefix=''):
    """Flatten nested config for MLflow parameter logging"""
    items = {}
    for key, value in config.items():
        if isinstance(value, dict):
            items.update(flatten_config_for_logging(value, f"{prefix}{key}."))
        elif isinstance(value, list):
            items[f"{prefix}{key}"] = ','.join(str(v) for v in value)
        elif isinstance(value, bool):
            items[f"{prefix}{key}"] = str(value).lower()
        else:
            items[f"{prefix}{key}"] = str(value)
    return items

def setup_mlflow(config, exp_dir, args):
    """Setup MLflow tracking"""
    mlflow_uri = config.get('mlflow_uri', 'file:./mlruns')
    mlflow.set_tracking_uri(mlflow_uri)
    
    experiment_name = config.get('experiment_name', 'nerf_experiments')
    mlflow.set_experiment(experiment_name)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset = config.get('dataset', 'unknown')
    run_name = f"{dataset}_{timestamp}"
    
    mlflow.start_run(run_name=run_name)
    
    # Log flattened config
    flat_config = flatten_config_for_logging(config)
    mlflow.log_params(flat_config)
    
    # Log config file
    mlflow.log_artifact(args.config)
    
    # Log saved config
    saved_config = exp_dir / 'config.yaml'
    if saved_config.exists():
        mlflow.log_artifact(saved_config)
    
    print(f"MLflow tracking started:")
    print(f"  Run: {run_name}")
    print(f"  Experiment: {experiment_name}")
    print(f"  URI: {mlflow_uri}")
    
    return mlflow.active_run()

def setup_experiment(config, args, device):
    """Setup experiment directory"""
    exp_name = config['experiment_name']
    basedir = config.get('basedir', './logs')
    exp_dir = Path(basedir) / exp_name
    ckpt_dir = exp_dir / 'checkpoints'

    if not exp_dir.exists():
        if args.eval:
            raise FileNotFoundError(f"Experiment directory {exp_dir} does not exist for evaluation.")
        exp_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        
        with open(exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Created experiment directory: {exp_dir}")
        resume_from = None
    else:
        print(f"Using existing experiment directory: {exp_dir}")
        resume_from = args.resume
        if not resume_from and not args.eval:
            ckpt_files = sorted(ckpt_dir.glob('*.ckpt'))
            if ckpt_files:
                latest_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
                resume_from = str(latest_ckpt)
                print(f"Auto-resuming from latest checkpoint: {resume_from}")
    
    return exp_dir, resume_from

def run_training(config, exp_dir, resume_from, device, args):
    """Run training"""
    
    # Get data paths
    data_config = config.get('data', {})
    data_path = data_config.get('train_path')
    if not data_path:
        raise ValueError("train_path must be specified in config data section")
    
    test_data_path = data_config.get('val_path') or data_config.get('test_path')
    if not test_data_path:
        # Try to infer
        train_path = Path(data_path)
        if 'train' in train_path.name:
            test_data_path = str(train_path.parent / train_path.name.replace('train', 'test'))
            print(f"Inferred test_data_path: {test_data_path}")
        else:
            test_data_path = data_path
    
    # Get model config
    model_config = config.get('model', {})
    model_kwargs = {
        'net_depth': model_config.get('net_depth', 8),
        'net_width': model_config.get('net_width', 256),
        'net_depth_fine': model_config.get('net_depth_fine', model_config.get('net_depth', 8)),
        'net_width_fine': model_config.get('net_width_fine', model_config.get('net_width', 256)),
        'N_samples': model_config.get('N_samples', 64),
        'N_importance': model_config.get('N_importance', 128),
        'use_viewdirs': model_config.get('use_viewdirs', True),
        'use_embed': model_config.get('use_embed', True),
        'multires': model_config.get('multires', 10),
        'multires_views': model_config.get('multires_views', 4),
        'ray_chunk': model_config.get('ray_chunk', 1024 * 32),
        'pts_chunk': model_config.get('pts_chunk', 1024 * 64),
        'perturb': 1.0,
        'raw_noise_std': 0.0,
        'white_bkgd': data_config.get('white_bkgd', False),
        'use_hierarchical': model_config.get('use_hierarchical', True)
    }
    
    # Get training config
    train_config = config.get('training', {})
    
    # Adjust for debug mode
    if args.debug:
        train_config['max_steps'] = 1000
        train_config['i_testset'] = 500
    
    # Call train function
    train(
        model_type=model_config.get('type', 'mipnerf'),
        data_path=data_path,
        test_data_path=test_data_path,
        run_dir=Path(exp_dir),
        batch_size=train_config.get('batch_size', 1024),
        max_steps=train_config.get('max_steps', 200000),
        lr=train_config.get('lr', 5e-4),
        lr_decay_steps=train_config.get('lr_decay_steps', 250000),
        lr_decay_rate=train_config.get('lr_decay_rate', 0.1),
        device=device,
        i_print=train_config.get('i_print', 100),
        i_weights=train_config.get('i_weights', 10000),
        i_testset=train_config.get('i_testset', 50000),
        log_img_idx=train_config.get('log_img_idx', 0),
        resume_from=resume_from,
        use_tensorboard=train_config.get('use_tensorboard', True),
        use_mlflow=args.mlflow,
        **model_kwargs
    )

def run_evaluation(config, exp_dir, device, resume_path):
    """Run evaluation"""
    print("EVALUATION MODE")
    
    ckpt_dir = exp_dir / 'checkpoints'
    ckpt_path = resume_path

    if not ckpt_path:
        if (ckpt_dir / 'final.ckpt').exists():
            ckpt_path = str(ckpt_dir / 'final.ckpt')
        else:
            ckpt_files = sorted(ckpt_dir.glob('*.ckpt'))
            if ckpt_files:
                ckpt_path = str(ckpt_files[-1])
    
    if not ckpt_path or not Path(ckpt_path).exists():
        raise ValueError(f"No checkpoint found for evaluation in {ckpt_dir}")

    print(f"Loading checkpoint: {ckpt_path}")
    
    # Get model config
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    model_kwargs = {
        'net_depth': model_config.get('net_depth', 8),
        'net_width': model_config.get('net_width', 256),
        'net_depth_fine': model_config.get('net_depth_fine', model_config.get('net_depth', 8)),
        'net_width_fine': model_config.get('net_width_fine', model_config.get('net_width', 256)),
        'N_samples': model_config.get('N_samples', 64),
        'N_importance': model_config.get('N_importance', 128),
        'use_viewdirs': model_config.get('use_viewdirs', True),
        'use_embed': model_config.get('use_embed', True),
        'multires': model_config.get('multires', 10),
        'multires_views': model_config.get('multires_views', 4),
        'ray_chunk': model_config.get('ray_chunk', 1024 * 32),
        'pts_chunk': model_config.get('pts_chunk', 1024 * 64),
        'perturb': 0.0,
        'raw_noise_std': 0.0,
        'white_bkgd': data_config.get('white_bkgd', False),
        'use_hierarchical': model_config.get('use_hierarchical', True)
    }
    
    model = MipNeRF(**model_kwargs).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    step = ckpt.get('step', 0)
    
    print(f"Evaluating checkpoint at step {step}")
    
    # Get test data path
    test_data_path = data_config.get('test_path') or data_config.get('val_path')
    if not test_data_path:
        train_path = data_config.get('train_path', '')
        if 'train' in train_path:
            test_data_path = train_path.replace('train', 'test')
        else:
            test_data_path = data_config.get('train_path')
    
    test_dataset = RayNeRFDataset(
        json_path=test_data_path,
        split='val',
        device=device
    )
    
    eval_config = config.get('evaluation', {})
    eval_dir = exp_dir / 'eval'
    evaluator = Eval(
        model=model,
        dataset=test_dataset,
        device=device,
        output_dir=eval_dir
    )
    
    results = evaluator.evaluate_all(
        max_images=eval_config.get('max_val_images', None),
        save_every=5,
        num_coarse=model_config.get('N_samples', 64),
        num_fine=model_config.get('N_importance', 128),
        use_viewdirs=model_config.get('use_viewdirs', True)
    )
    
    print(f"Evaluation complete. Results saved to {eval_dir}")
    return results

def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup experiment
    exp_dir, resume_from = setup_experiment(config, args, device)
    
    # Setup MLflow if enabled for training
    if args.mlflow and not args.eval:
        mlflow_run = setup_mlflow(config, exp_dir, args)
    
    try:
        if args.eval:
            results = run_evaluation(config, exp_dir, device, args.resume or resume_from)
            
            # Log evaluation to MLflow if enabled
            if args.mlflow:
                mlflow_uri = config.get('mlflow_uri', 'file:./mlruns')
                mlflow.set_tracking_uri(mlflow_uri)
                mlflow.set_experiment(config.get('experiment_name', 'nerf_experiments'))
                
                eval_run_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                mlflow.start_run(run_name=eval_run_name)
                
                if results:
                    mlflow.log_metrics({
                        'psnr_mean': results.get('psnr', {}).get('mean', 0),
                        'ssim_mean': results.get('ssim', {}).get('mean', 0),
                        'lpips_mean': results.get('lpips', {}).get('mean', 0)
                    })
                
                mlflow.end_run()
        else:
            run_training(config, exp_dir, resume_from, device, args)
    finally:
        if args.mlflow and not args.eval and mlflow.active_run():
            mlflow.end_run()
            print("MLflow run ended")

if __name__ == '__main__':
    main()