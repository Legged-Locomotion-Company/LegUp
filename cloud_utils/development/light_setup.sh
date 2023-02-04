cd /root;

echo "downloading apt packages";

apt-get update -qq;
apt-get install -y -qq\
    build-essential \
    python3.8 \
    python3-pip \
    libpython3.8 \
    libpython3.8-dev \
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
    mesa-vulkan-drivers \
    pigz \
    git \
    libegl1 \
    libgl1 \
    git-lfs \
    megatools;

update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 0;
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8;
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8;

export NVIDIA_VISIBLE_DEVICES=all;
export NVIDIA_DRIVER_CAPABILITIES=all;

echo "creating conda env";

conda create --name legsenv python=3.8;

source /opt/conda/etc/profile.d

conda activate legsenv;

pip3 install --upgrade pip --quiet;

echo "grabbing isaacgym"

megadl 'https://mega.co.nz/#!6f4jnDqB!b_z5kvu8yfmmRdvxfAbSgXa69QSAOcWlkiMCEFTxJ6M';
tar -xzf /root/IsaacGym_Preview_4_Package.tar.gz;

cp ./isaacgym/docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
cp ./isaacgym/docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

echo "installing isaacgym"

pip install -q -e /root/isaacgym/python/.;

git clone https://AndrewMead10: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@github.com/Legged-Locomotion-Company/LegUp;

cd /root/LegUp;

git checkout mish_branch;

echo "installing legup"

pip3 install --no-input -r requirements.txt --quiet;
pip3 install --no-input -e . --quiet;

echo "installing nvitop"

pip3 install nvitop;

wandb login ab4474327941fce151a79358a59ee85a3377e660

echo "SETUP DONE"