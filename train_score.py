from score_models import NCSNpp, ScoreModel
from torchvision import transforms as T
import torch
import json
import h5py

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, channels=2, condition_on_z=False, condition_on_sed=False, condition_on_mag=False):
        self.file = h5py.File(path)
        self.len = len(self.file["hudf_resized"])
        self.channels = channels
        self.condition_on_z = condition_on_z
        self.condition_on_mag = condition_on_mag
        self.condition_on_sed = condition_on_sed

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        image = torch.tensor(self.file["hudf_resized"][index, self.channels].astype(np.float32)).to(DEVICE)
        if image.ndim == 2:
            image = image[None]
        args = []
        if self.conditioned_on_z:
            z = torch.tensor(self.file["hudf_z"][index].astype(np.float32)).to(DEVICE)
            args.append(z)
        if self.condition_on_sed
            sed = torch.tensor(self.file["hudf_template"][index].astype(np.float32)).to(DEVICE)
            args.append(sed)
        return image, *args
            

def main(args):
    with open(args.parameter_path, "r") as f:
        hp = json.load(f)

    transformation_sequence= T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        ])
    def preprocessing(image):
        image = transformation_sequence(image)
        return image
    
    net = NCSNpp(**hp)
    model = ScoreModel(net, **hp)
    dataset = Dataset(args.dataset_path, args.channels)
    model.fit(
            dataset,
            preprocessing_fn=preprocessing,
            epochs=args.epochs,
            batch_size=args.batch_size,
            checkpoints_directory=args.checkpoints_directory,
            learning_rate=args.learning_rate
            )

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_argument("--dataset_path", required=True, help="Path to the h5 dataset")
    parser.add_argument("--channels", default=2, nargs="+" type=int, help="Channels to train the model on. Multiple channel can be provided")
    parser.add_argument("--parameter_path", required=True, help="Path to the model json hyperparameter file")
    parser.add_argument("--checkpoints_directory", required=True, help="Path to the folder where to save the model, created if it does not exist.")
    
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--max_time", default=np.inf, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--condition_on_z", action="store_true")
    parser.add_argument("--condition_on_sed", action="store_true")
    args = parser.parse_args()
    main()
