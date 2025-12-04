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
echo "⠰⠿⠿⠿⠿⠋⠠⠾⠿⠿⠋ Nerf-GPU ⠹⣿⣿⣷⠄⠙⠿⠿⠿⠿⠆"
echo ""

cd /workspace
git clone https://github.com/forSplinter/NeRFs-Arch.git
cd NeRFs-Arch

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

uv venv .venv
source .venv/bin/activate

uv pip install e . --no-deps
uv pip install mlflow numpy tqdm pillow scikit-image opencv-python imageio pyyaml
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db &
sleep 5

echo "Training started..."
nerf --config configs/gpu/lot1_gpu.yaml --mlflow

echo "Training complete."