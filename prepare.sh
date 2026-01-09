conda create -n rekv python=3.11 -y
conda activate rekv

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

git clone https://gitee.com/ahead5822/transformers.git
cd transformers
git checkout 66bc4def9505fa7c7fe4aa7a248c34a026bb552b
pip install -e .

cd ..
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 
cd Rekv
pip install -e .

cd model/longva
pip install -e .