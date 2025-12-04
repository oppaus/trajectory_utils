import unittest
import numpy as np
import torch.nn as nn
import torch
import os
from typing import List
from trajectory_metrics import TrajectoryMetrics, TrajMetric, load_checkpoint
from train import load_dataset
from torch.utils.data import TensorDataset, DataLoader, random_split

class MyTestCase(unittest.TestCase):
    def test_metrics_rand(self):
        # construct metrics from random seeds; run on one chkpt file
        model = load_checkpoint("trajectories_big_1_32", [205,])

        # construct metrics
        tmet = TrajectoryMetrics()
        tmet.reset_random_batches(
            state_vec=np.array([0.0, np.pi/4, 0.0, 0.0]),
            num_batches=10,
            batch_size=128,
            seed=42)
        tmet.reset_accumulators()

        metrics = tmet.eval_metrics_rand(model, device=torch.device("cpu"))

        print("\n")
        metrics.pretty_print()

    def test_metrics_loader(self):
        # construct metrics from existing stored s0; run on one chkpt file
        model = load_checkpoint("trajectories_big_1_32", [205,])

        # load s0 from training dataset
        __, train_s0, _, _ = load_dataset("trajectories_big_1_training.npz")
        train_s0_loader = DataLoader(train_s0, batch_size=128, shuffle=False)
        # construct metrics
        tmet = TrajectoryMetrics()
        tmet.reset_accumulators()

        metrics = tmet.eval_metrics(model, train_s0_loader, device=torch.device("cpu"))
        print("\n")
        metrics.pretty_print()

    def test_unicode(self):
        # sigh - I give up on \dot{\theta}. The unicode for this
        # just merges the tiny little dot over \theta into the \theta.
        # Switched to ' for time derivative...
        description = "metrics"
        symbols = ["x", "θ", "x'", "θ'"]
        vals = np.array([0.12, 0.34, 0.56, 0.78])

        line = (
                f"{description} | "
                + " | ".join(f"{s}: {v:.4f}" for s, v in zip(symbols, vals))
                + " |"
        )
        print("\n")
        print(line)


if __name__ == '__main__':
    unittest.main()
