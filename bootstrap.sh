#!/bin/bash
# bootstrap.sh — sets up a fresh Vast.ai machine for SAGE
# Usage: bash bootstrap.sh
set -e  # exit on first error

echo "=== [1/6] System packages ==="
apt-get update
apt-get install -y ffmpeg git-lfs wget

echo "=== [2/6] Miniconda (if not present) ==="
if ! command -v conda &> /dev/null; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
else
    eval "$(conda shell.bash hook)"
fi

# Accept Anaconda ToS to avoid the prompt
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo "=== [3/6] Conda env 'sage' ==="
if ! conda env list | grep -q "^sage "; then
    conda create --name sage -y python=3.11
fi
conda activate sage

echo "=== [4/6] Python deps ==="
pip install --upgrade pip
pip install decord qwen_vl_utils
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e .
pip install -e verl/
pip install transformers==4.57.0
pip install vllm==0.11.0
pip install flash-attn==2.7.3 --no-build-isolation
# pip install trl deepspeed  # uncomment only if you need SFT

echo "=== [5/6] Tools for data download ==="
pip install -U yt-dlp huggingface_hub pandas pyarrow

echo "=== [6/6] HuggingFace cache hint ==="
echo "If you have a persistent volume mounted, set HF_HOME to point there so model weights survive instance restarts:"
echo "  export HF_HOME=/workspace/hf-cache"

echo ""
echo " Bootstrap complete. Run: conda activate sage"
