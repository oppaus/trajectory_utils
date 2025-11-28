# nn model training

import numpy as np
import sklearn.utils
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os

def eval_loss(model: nn.Sequential, loader: DataLoader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            # Weight by batch size to get dataset-wide mean later
            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
    return total_loss / n_samples

def train(dname: str, restart_epoch: int = -1):
    """
    dname: directory name for training
    restart_epoch: if > 0, used to reload
        previous model weights and continue training
    """
    # load data into dataset
    fname = dname + ".npz"
    data = np.load(fname)
    x = data['first_array']
    y = data['second_array']
    print(x.shape,y.shape)

    # shuffle these before splitting into train and validation sets
    x, y = sklearn.utils.shuffle(x, y, random_state=42)

    xt = torch.from_numpy(x).float()
    yt = torch.from_numpy(y).float()

    full_dataset = TensorDataset(xt, yt)

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

    input_dim = x.shape[1]
    output_dim = y.shape[1] if y.ndim > 1 else 1
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

    train_dir = os.path.join("train",dname)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 400

    train_loss_vec = []
    val_loss_vec = []
    epoch_vec = []

    for epoch in range(restart_epoch, restart_epoch + num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

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
                f"Epoch {epoch + 1:3d}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f} "
            )

            train_loss_vec.append(train_loss)
            val_loss_vec.append(val_loss)
            epoch_vec.append(epoch)
            checkpoint = {"model_state": model.state_dict(),
                          "optimizer_state": optimizer.state_dict(),
                          "input_dim": input_dim,
                          "output_dim": output_dim,
                          "hidden_dim": hidden_dim,
                          "dropout_rate": dropout_rate}
            cname = dname + "_" + str(restart_epoch) + "_" + str(epoch) + ".pth"
            cpath = os.path.join(train_dir, cname)
            torch.save(checkpoint, cpath)

    plt.figure()
    plt.plot(epoch_vec, train_loss_vec, 'r')
    plt.plot(epoch_vec, val_loss_vec, 'g')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    fname = "learning_curves_" + str(restart_epoch) + ".png"
    lcname = os.path.join(train_dir, fname)
    plt.savefig(lcname)
    plt.show()

if __name__ == "__main__":
    train(dname="trajectories_big_1", restart_epoch=0)
