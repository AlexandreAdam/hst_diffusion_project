from score_models import NCSNpp, ScoreModel
from torchvision improt transforms as T
import torch
import json
import h5py

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, channel=2):
        self.file = h5py.File(path)
        self.len = len(self.file["hudf_resized"])
        self.channel = channel

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        image = torch.tensor(self.file["hudf_resized"][index, self.channel].astype(np.float32)).to(DEVICE)[None]
        # z = ...
        return image


def preprocessing(image):
    # flips
    # rotation
    return image


def main(args):
    with open(args.parameter_path, "r") as f:
        hp = json.load(f)
    
    
    net = NCSNpp(**hp)
    model = ScoreModel(net, **hp)
    dataset = Dataset(args.dataset_path, args.channel)
    model.fit(
            dataset,
            preprocessing_fn=preprocessing,
            epochs=
            batch_size=,
            checkpoint_directory=
            )

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--channel", default=2, type=int)
    parser.add_argument("--parameter_path", required=True)
    args = parser.parse_args()
    main()
