"""Compare loss between torch/pth and onnxruntime/onnx across checkpoints.
Using checkpoints from a torch training set, create onnx exports as needed then
run inference through torch/onnxruntime and compute loss. 
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import onnxruntime as ort
import sklearn.utils
import numpy as np

class Modelizer(object):
    def __init__(self, basepath: str):
        # path string to checkpoint file (minus extension)
        # example: train/trajectories_big_1/trajectories_big_1_0_0
        self.basepath = basepath
        self.init()

    def init(self):
        """Load (from file) model and get it ready to predict."""
        raise NotImplementedError("Oops! Naughty subclass must define.")

    def predict(self, states: TensorDataset):
        """Run the model prediction
        Args:
            states (TensorDataset): model input, represents the system states
        Returns:
            TODO
        """
        raise NotImplementedError("Oops! Naughty subclass must define.")

class TorchModel(Modelizer):
    # TODO: profile this method, very slow on first initialization, might just
    # be torch loading for the first time, subsequents calls seem fine.
    def init(self):
        # TODO: workaround, see other TODO in eval_loss function below
        self.device = torch.device("cpu")
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fpath = self.basepath + ".pth"

        # using weights only is much faster (and safer)
        ckpt = torch.load(fpath, map_location="cpu", weights_only=True) 

        # extract relevant dimensional values
        input_dim, hidden_dim, output_dim, dropout_rate = [ckpt[k] for k in
            ["input_dim","hidden_dim","output_dim","dropout_rate"]]
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.model.load_state_dict(ckpt["model_state"])

        # prep for inference (turn off Dropout, engage consistent BatchNorm)
        self.model.eval()
        self.model.to(self.device)

    def predict(self, states):
        with torch.no_grad():
            states = states.to(self.device)
            return self.model(states)

class OnnxModel(Modelizer):
    def init(self):
        fpath = self.basepath + ".onnx"

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"{self.basepath}")

        self.sess = ort.InferenceSession(fpath)

    def predict(self, states):
        results = self.sess.run(None,{'input': states.numpy()})[0]
        return torch.from_numpy(results)

def eval_loss(model: Modelizer, loader: DataLoader, loss_fn):
    """Evaluate a model using the given loss function.
    """
    total_loss = 0.0
    n_samples = 0
    for xb, yb in loader:
        preds = model.predict(xb)
        # TODO: torch chokes if using GPU and yb isn't transferred there, fine
        # in ONNXRuntime, short-term fix is to run torch inference on CPU
        loss = loss_fn(preds, yb)
        # Weight by batch size to get dataset-wide mean later
        batch_size = xb.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size
    return total_loss / n_samples

def load_training_data(fname: str) -> TensorDataset:
    """Loads training data from the given numpy zip archive.
    Args:
        fname (str): path to an npzfile
    Returns:
        TensorDataset: contents of the npzfile
    """
    data = np.load(fname)

    # States (input)
    x = data['first_array']

    # Control (output)
    y = data['second_array']

    # shuffle these before splitting into train and validation sets
    x, y = sklearn.utils.shuffle(x, y, random_state=42)

    xt = torch.from_numpy(x).float()
    yt = torch.from_numpy(y).float()

    return TensorDataset(xt, yt)

def split_training_data(full_dataset: TensorDataset):
    """Split dataset into training/validation.
    Args:
        full_dataset (TensorDataset): the entire training dataset 
    Returns:
        tuple: (DataLoader: training, DataLoader: validation)
    """
    train_frac = 0.8
    train_size = int(train_frac * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # reproducible split
    )

    batch_size = 128
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return (train_loader, val_loader)


