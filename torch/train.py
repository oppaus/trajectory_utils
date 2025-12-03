# nn model training

import numpy as np
import sklearn.utils
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from trajectory_metrics import TrajectoryMetrics, TrajMetric
from sklearn.utils import shuffle
from typing import Tuple
import os

def eval_loss(model: nn.Sequential, loader: DataLoader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    p = next(model.parameters())
    with torch.no_grad():
        # iterate over batches
        for xb, yb in loader:
            xb = xb.to(dtype=p.dtype, device=p.device)
            yb = yb.to(dtype=p.dtype, device=p.device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            # Weight by batch size to get dataset-wide mean later
            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
    return total_loss / n_samples

def load_dataset(fname: str) -> Tuple[TensorDataset, TensorDataset, int, int]:
    data = np.load(fname)
    x = data['first_array']
    y = data['second_array']
    z = data['third_array']

    print(x.shape, y.shape, z.shape)

    # shuffle nn data before splitting into train and validation sets
    x, y = sklearn.utils.shuffle(x, y, random_state=42)

    input_dim = x.shape[1]
    output_dim = y.shape[1] if y.ndim > 1 else 1

    xt = torch.from_numpy(x).float()
    yt = torch.from_numpy(y).float()
    zt = torch.from_numpy(z).float()

    dataset = TensorDataset(xt, yt)
    metric_dataset = TensorDataset(zt)
    return dataset, metric_dataset, input_dim, output_dim

def train(dname: str, output_dir: str, num_epochs: int, model_type: torch.dtype, restart_epoch: int = -1):
    """
    dname: directory name for training
    restart_epoch: if > 0, used to reload
        previous model weights and continue training
    """
    # load data - already split
    fname = dname + "_training.npz"
    train_ds, train_s0, input_dim, output_dim = load_dataset(fname)
    fname = dname + "_validation.npz"
    val_ds, val_s0, _, _ = load_dataset(fname)

    batch_size = 128
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_s0_loader = DataLoader(train_s0, batch_size=batch_size, shuffle=False)
    val_s0_loader = DataLoader(val_s0, batch_size=batch_size, shuffle=False)

    tm = TrajectoryMetrics()

    hidden_dim = 32
    dropout_rate = 0.0

    model = nn.Sequential(
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

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = model.to(device=device, dtype=model_type)

    train_dir = os.path.join("train",output_dir)
    os.makedirs(train_dir, exist_ok=True)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if restart_epoch > 0:
        fname = dname + "_" + str(restart_epoch) + ".pth"
        fpath = os.path.join(train_dir, fname)
        ckpt = torch.load(fpath, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        restart_epoch = restart_epoch + 1
    else:
        restart_epoch = 0

    train_loss_vec = []
    val_loss_vec = []
    epoch_vec = []
    train_overall_success_rate = []
    val_overall_success_rate = []

    p = next(model.parameters())

    for epoch in range(restart_epoch, restart_epoch + num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(dtype=p.dtype, device=p.device)
            yb = yb.to(dtype=p.dtype, device=p.device)

            # forward
            pred = model(xb)
            loss = loss_fn(pred, yb)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if epoch % 5 == 0:
            train_loss = eval_loss(model, train_loader, loss_fn, device)
            val_loss = eval_loss(model, val_loader, loss_fn, device)

            print(
                f"Epoch {epoch:3d}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f} "
            )

            tm.reset_accumulators()
            train_metrics = tm.eval_metrics(model, train_s0_loader, device)
            tm.reset_accumulators()
            val_metrics = tm.eval_metrics(model, val_s0_loader, device)
            print(f"Epoch {epoch:3d}: Training metrics:")
            train_metrics.pretty_print()
            print(f"Epoch {epoch:3d}: Validation metrics:")
            val_metrics.pretty_print()

            train_loss_vec.append(train_loss)
            val_loss_vec.append(val_loss)
            epoch_vec.append(epoch)
            train_overall_success_rate.append(train_metrics.all_success)
            val_overall_success_rate.append(val_metrics.all_success)

            checkpoint = {"model_state": model.state_dict(),
                          "optimizer_state": optimizer.state_dict(),
                          "input_dim": input_dim,
                          "output_dim": output_dim,
                          "hidden_dim": hidden_dim,
                          "dropout_rate": dropout_rate}
            cname = dname + "_" + str(epoch) + ".pth"
            cpath = os.path.join(train_dir, cname)
            torch.save(checkpoint, cpath)

    plt.figure()
    plt.plot(epoch_vec, train_loss_vec, 'r', label="training")
    plt.plot(epoch_vec, val_loss_vec, 'g', label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.title("Learning curves: loss")
    fname = "learning_curves_" + str(restart_epoch) + ".png"
    lcname = os.path.join(train_dir, fname)
    plt.savefig(lcname)
    plt.show()

    plt.figure()
    plt.plot(epoch_vec, train_overall_success_rate, 'r', label="training")
    plt.plot(epoch_vec, val_overall_success_rate, 'g', label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("All success rate")
    plt.legend()
    plt.title("Learning curves: metrics")
    fname = "metric_curves_" + str(restart_epoch) + ".png"
    lcname = os.path.join(train_dir, fname)
    plt.savefig(lcname)
    plt.show()

if __name__ == "__main__":
    train(dname="trajectories_big_1", output_dir="trajectories_big_1_64", model_type=torch.float64, num_epochs=400, restart_epoch=-1)
