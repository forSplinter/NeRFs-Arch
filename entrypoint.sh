#!/bin/bash
echo ""
echo "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀"
echo "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀"
echo "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠈⢻⣿⣿⠟⢀⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀"
echo "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢴⣿⣷⡄⠹⠃⢠⣾⣿⡦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀"
echo "⠀⠀⠀⠀⠀⠀⠀⠀⢠⣦⡀⠻⣿⠟⠀⠀⢻⣿⠟⠁⣴⡄⠀⠀⠀⠀⠀⠀⠀⠀"
echo "⠀⠀⠀⠀⠀⠀⠀⠐⢿⣿⣷⣄⠉⢀⣾⣧⡀⠉⢠⣾⣿⡿⠂⠀⠀⠀⠀⠀⠀⠀"
echo "⠀⠀⠀⠀⠀⢀⣾⣧⡀⠻⡿⠁⣠⠈⢻⡟⠁⣀⠈⢿⠟⢀⣼⣷⡀⠀⠀⠀⠀⠀"
echo "⠀⠀⠀⠀⣀⠈⢿⣿⣷⠄⠀⢾⣿⣷⠄⠀⣾⣿⡦⠀⠠⣾⣿⡟⠁⣄⠀⠀⠀⠀"
echo "⠀⠀⢀⣼⣿⣧⡀⠙⠃⣠⣦⠈⠻⠁⠀⠀⠘⠟⠁⣴⣄⠙⠋⢀⣾⣿⣧⡀⠀⠀"
echo "⠀⣠⣿⣿⣿⣿⠟⢀⣼⣿⣿⠗⠀⠀⠀⠀⠀⠀⠺⣿⣿⣦⡀⢻⣿⣿⣿⣿⣄⠀"
echo "⠰⠿⠿⠿⠿⠋⠠⠾⠿⠿⠋⠀⠀⠀ NeRF x4 ⠀⠀⠹⣿⣿⣷⠄⠙⠿⠿⠿⠿⠆"
echo ""

cd /workspace
git clone https://github.com/forSplinter/NeRFs-Arch.git
cd NeRFs-Arch

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

uv venv .venv
source .venv/bin/activate
uv pip install -e .
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

cat > configs/lot1_gpu.yaml << 'EOF'
experiment_name: lot1_4x3090
dataset: lot1
mlflow_uri: "http://0.0.0.0:5001"

data:
  train_path: "nerfdataset/lot1/transforms_train.json"
  val_path: "nerfdataset/lot1/transforms_val.json"
  test_path: "nerfdataset/lot1/transforms_test.json"
  white_bkgd: true
  near: 0.1
  far: 100.0

model:
  type: "mipnerf"
  net_depth: 8
  net_width: 256
  net_depth_fine: 8
  net_width_fine: 256
  N_samples: 128
  N_importance: 256
  use_viewdirs: true
  use_embed: true
  multires: 10
  multires_views: 4
  white_bkgd: true
  use_hierarchical: true

training:
  batch_size: 16384  # Énorme avec 4 GPUs !
  max_steps: 200000
  lr: 5e-4
  lr_decay_steps: 150000
  lr_decay_rate: 0.1
  i_print: 100
  i_weights: 10000
  i_testset: 50000
  num_workers: 16

evaluation:
  save_images: false
  save_comparisons: false
  max_val_images: 5
  metrics: ["psnr", "ssim", "lpips"]

system:
  seed: 42
  precision: "fp32"
  benchmark: true
  deterministic: false
  device: "cuda"
EOF

mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db &
sleep 5

echo "Lancement sur 4x RTX 3090..."
nerf --config configs/lot1_gpu.yaml --mlflow

echo " Entraînement terminé !"
echo " MLflow: http://38.22.92.210:5001"