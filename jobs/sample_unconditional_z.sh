#!/bin/bash
#SBATCH --array=1-10%2
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G                        # memory per node
#SBATCH --time=00-05:00         # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Sample_uncond_z
#SBATCH --output=%x-%j.out

base_dir=$HOME/scratch/hst_diffusion_project

source $HOME/environments/milex/bin/activate
python $base_dir/unconditional_sampling.py\
    --checkpoints_directory=$base_dir/ncsnpp_hst_z_unconditional\
    --batch_size=10\
    --output_directory=$base_dir/samples/\
    --name_prefix=samples_unconditional_z\
    --total=10000\
    --em_steps=1000\

