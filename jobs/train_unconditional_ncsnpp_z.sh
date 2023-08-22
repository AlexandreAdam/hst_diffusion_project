#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G                        # memory per node
#SBATCH --time=02-00:00         # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=TrainHST_z
#SBATCH --output=%x-%j.out

source $HOME/environments/milex/bin/activate
python $HOME/scratch/hst_diffusion_project/train_score.py\
    --dataset_path=$HOME/scratch/hst_diffusion_project/data/hudf_dataset.h5\
    --channels 3\
    --parameter_path=$HOME/scratch/hst_diffusion_project/architectures/unconditional_ncsnpp.json\
    --checkpoints_directory=$HOME/scratch/hst_diffusion_project/ncsnpp_hst_z_unconditional\
    --batch_size=4\
    --epochs=10000\
    --max_time=47\

