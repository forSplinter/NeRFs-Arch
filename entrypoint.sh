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

echo "[1/6] Installing dependencies..."
apt-get update
apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt install -y python3.11 python3.11-venv python3.11-dev
python3.11 --version || echo "Python installation failed"

echo "[2/6] Setting up workspace..."
cd /workspace
if [ -d "NeRFs-Arch" ]; then
    echo "NeRFs-Arch directory already exists. Removing it for a fresh clone."
    cd NerFs-Arch
    git pull
else 
    git clone https://github.com/forSplinter/NeRFs-Arch.git
    cd NeRFs-Arch
fi

echo "[3/6] Installing UV and project dependencies..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

echo "[4/6] Setting up virtual environment..."
uv venv .venv
source .venv/bin/activate
python --version || echo "Virtual environment setup failed"

echo "[5/6] Installing dependencies in virtual environment..."
uv pip install -r pyproject.toml

echo "[6/6] Starting MLflow server and training..."
mlflow server --host 0.0.0.0 --port 8080 --backend-store-uri sqlite:///mlflow.db &
sleep 5
echo "Checking for CUDA availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo "Setup complete !!"