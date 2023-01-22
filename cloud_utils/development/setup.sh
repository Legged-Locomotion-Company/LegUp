sudo apt-get update;
sudo apt-get install -y --no-install-recommends \
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
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8;
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8;
sudo apt install megatools;
export NVIDIA_VISIBLE_DEVICES=all;
export NVIDIA_DRIVER_CAPABILITIES=all;
export PATH="/home/legsuser/.local/bin:$PATH"
megadl 'https://mega.co.nz/#!6f4jnDqB!b_z5kvu8yfmmRdvxfAbSgXa69QSAOcWlkiMCEFTxJ6M';
tar -xvzf IsaacGym_Preview_4_Package.tar.gz;
rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json;
cp ./isaacgym/docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json;
cp ./isaacgym/docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json;
cd isaacgym/python;
pip3 install -e .;
cd ../..;
sudo cp /opt/conda/lib/libpython3.8.so.1.0 /usr/lib/
git clone https://AndrewMead10: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@github.com/Legged-Locomotion-Company/LegUp;
cd LegUp;
pip install -e .;
cd Legup;
pip install -r requirements.txt;
wandb login aa97ba8812f4389ad4905d767febba2688c59671;
