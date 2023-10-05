from score_models import ScoreModel
import numpy as np
import torch
import h5py
import os
from scipy.stats import gaussian_kde

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))

def main(args):
    hf = h5py.File(args.dataset_path, "r")
    model = ScoreModel(checkpoints_directory=args.checkpoints_directory)
    
    # First, construct empirical distributions for sed and z based on dataset
    bins = np.arange(101) / 10 + 1
    tokenize = lambda sed: np.digitize(sed, bins)
    seds, sed_counts = np.unique(hf["hudf_template"][:], return_counts=True)
    sed_p = sed_counts / sed_counts.sum()
    sed_dist = lambda n: torch.tensor(tokenize(np.random.choice(seds, p=sed_p, size=n))).long().to(DEVICE)
    
    kde = gaussian_kde(hf["hudf_z"][:].astype(np.float32))
    z_dist = lambda n: torch.tensor(kde.resample(n).flatten()).float().to(DEVICE)
   
    N = args.total
    B = args.batch_size
    with h5py.File(os.path.join(args.output_directory, args.name_prefix + f"_{THIS_WORKER}.h5"), "w") as hf:
        hf.create_dataset("images", [N//N_WORKERS, args.channels, args.pixels, args.pixels], dtype=np.float32)
        hf.create_dataset("z", [N//N_WORKERS], dtype=np.float32)
        hf.create_dataset("sed", [N//N_WORKERS], dtype=np.int32)
        for b in range(N // N_WORKERS // B):
            z = z_dist(B)
            sed = sed_dist(B)
            samples = model.sample([B, args.channels, args.pixels, args.pixels], args.em_steps, condition=[z, sed])
            hf["images"][b*B: (b+1)*B] = samples.cpu().numpy().astype(np.float32)
            hf["z"][b*B: (b+1)*B] = z.cpu().numpy().astype(np.float32)
            hf["sed"][b*B: (b+1)*B] = sed.cpu().squeeze().numpy().astype(np.int32)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
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
