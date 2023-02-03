cd /root;

apt-get update;
apt-get install -y \
    build-essential \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    mesa-common-dev \
    zip \
    unzip \
    make \
    gcc-8 \
    g++-8 \
    vulkan-utils \
    pigz \
    git \
    libegl1 \
    git-lfs \
    megatools;

megadl 'https://mega.co.nz/#!6f4jnDqB!b_z5kvu8yfmmRdvxfAbSgXa69QSAOcWlkiMCEFTxJ6M';
tar -xvzf /root/IsaacGym_Preview_4_Package.tar.gz;

conda init bash;

conda create -y --name legenv python=3.8;
source /opt/conda/etc/profile.d/conda.sh; 
conda activate legenv;

cp /opt/conda/envs/legenv/lib/libpython3.8.so.1.0 /usr/lib/;

conda info;

cd isaacgym/python;
pip3 install --no-input -e .;

cd ../..;

git clone https://AndrewMead10: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@github.com/Legged-Locomotion-Company/LegUp;

cd LegUp;

git checkout mish_branch;

pip3 install --no-input -e .;
pip3 install --no-input -r requirements.txt;

pip3 install nvitop;

wandb login ab4474327941fce151a79358a59ee85a3377e660

echo SETUP DONE