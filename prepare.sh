#!/usr/bin/env bash
# StreamKV environment setup — source of truth for the `streamkv` conda env.
#
# Sets up:
#   - the `streamkv` env (Python 3.11)
#   - torch 2.6.0 +cu124
#   - the custom transformers fork pinned to commit 66bc4def (the StreamKV monkey-patches
#     target internal HF attention APIs, so a stock transformers at a different commit will
#     likely break)
#   - the ABI-matched flash-attn prebuilt wheel for torch 2.6.0+cu124 (OLD C++11 ABI)
#
# The torch pin and the flash-attn wheel are load-bearing: the transformers commit + the
# monkey-patches were tested on torch 2.6.0, and the abiFALSE flash-attn wheel is the one that
# matches the cu124 torch build. Don't unpin without re-verifying.
set -euo pipefail

conda create -n streamkv python=3.11 -y
conda activate streamkv

# torch 2.6.0 +cu124
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

# transformers pinned to 66bc4def
pip install "git+https://github.com/huggingface/transformers.git@66bc4def9505fa7c7fe4aa7a248c34a026bb552b"

# flash-attn prebuilt wheel (ABI-matched for torch 2.6.0+cu124 OLD C++ ABI)
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# StreamKV root package (also pulls accelerate, av, decord, etc. from pyproject.toml)
pip install -e .
