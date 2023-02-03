sudo apt-get install megatools


megadl 'https://mega.co.nz/#!6f4jnDqB!b_z5kvu8yfmmRdvxfAbSgXa69QSAOcWlkiMCEFTxJ6M';
tar -xvzf IsaacGym_Preview_4_Package.tar.gz;

conda create --name legenv python=3.8;
conda activate legenv;

cd isaacgym/python;
pip3 install -e .;

cd ../..;

git clone https://AndrewMead10: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@github.com/Legged-Locomotion-Company/LegUp;

cd LegUp;
pip install -e .;
pip install -r requirements.txt;

wandb login ab4474327941fce151a79358a59ee85a3377e660