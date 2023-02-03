cd /root;

sudo apt-get update;
sudo apt-get install -y \
    megatools \
    nvtop;

megadl 'https://mega.co.nz/#!6f4jnDqB!b_z5kvu8yfmmRdvxfAbSgXa69QSAOcWlkiMCEFTxJ6M';
tar -xvzf IsaacGym_Preview_4_Package.tar.gz;

conda create -y --name legenv python=3.8;
source /opt/conda/etc/profile.d/conda.sh; 
conda activate legenv;

cd isaacgym/python;
pip3 install --no-input -e .;

cd ../..;

git clone https://AndrewMead10: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@github.com/Legged-Locomotion-Company/LegUp;

cd LegUp;
pip install --no-input -e . -;
pip install --no-input -r requirements.txt;

wandb login ab4474327941fce151a79358a59ee85a3377e660

echo SETUP DONE