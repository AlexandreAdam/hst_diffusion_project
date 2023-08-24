from score_models import ScoreModel
import numpy as np
import torch
import h5py
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))

def main(args):
    model = ScoreModel(checkpoints_directory=args.checkpoints_directory)
   
    N = args.total
    B = args.batch_size
    with h5py.File(os.path.join(args.output_directory, args.name_prefix + f"_{THIS_WORKER}.h5"), "w") as hf:
        hf.create_dataset("images", [N//N_WORKERS, args.channels, args.pixels, args.pixels], dtype=np.float32)
        for b in range(N // N_WORKERS // B):
            samples = model.sample(shape=[B, args.channels, args.pixels, args.pixels], steps=args.em_steps)
            hf["images"][b*B: (b+1)*B] = samples.cpu().numpy().astype(np.float32)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--checkpoints_directory", required=True, help="Path to the folder where to save the model, created if it does not exist.")
    parser.add_argument("--channels", default=1, type=int)
    parser.add_argument("--pixels", default=224, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--output_directory", required=True)
    parser.add_argument("--name_prefix", required=True)
    parser.add_argument("--em_steps", default=1000, type=int)
    parser.add_argument("--total", required=True, type=int)
    args = parser.parse_args()
    main(args)
