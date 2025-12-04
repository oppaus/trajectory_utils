""" Metrics for rollout trajectories.
These metrics assume a set of initial states for the data being
fed to them. The model is evaluated across all batches by rolling out
from these initial conditions. """

import torch
import os
import glob
import torch.nn as nn
import numpy as np
from scp_solver import SolverParams
from cartpole_solver_velocity import CartpoleSolverVelocity, CartpoleEnvironmentParams
from trajectory import TrajectoryScenario, Trajectory
from plot_trajectory import plot_trajectory
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import Tensor
from dataclasses import dataclass
from typing import List, Tuple

def load_checkpoint(trained_model_dir: str, trained_model_epoch: List[int]) -> nn.Sequential:
    tdir = trained_model_dir
    epoch_num = trained_model_epoch
    fdir = os.path.join("train",tdir)
    epoch_str = ""
    for sub_epoch in trained_model_epoch:
        epoch_str += "_"
        epoch_str += str(sub_epoch)
    fname = "*" + epoch_str + ".pth"
    fpath = os.path.join(fdir, fname)
    matches = glob.glob(fpath)
    if len(matches) == 0:
        raise FileNotFoundError("No files matched the pattern")
    elif len(matches) > 1:
        raise RuntimeError(f"Multiple files matched: {matches}")
    path = matches[0]
    ckpt = torch.load(path, map_location="cpu")

    input_dim = ckpt["input_dim"]
    hidden_dim = ckpt["hidden_dim"]
    output_dim = ckpt["output_dim"]
    dropout_rate = ckpt["dropout_rate"]

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

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def to_nn_state(s: np.ndarray, norm: list):
    #in_shape = s.shape
    #snn = np.zeros((in_shape[0], 4))
    sn = s / norm
    snn = sn
    # snn[:, 0] = sn[:, 0]
    # snn[:, 1] = np.cos(sn[:, 1])
    # snn[:, 2] = np.sin(sn[:, 1])
    # snn[:, 3] = sn[:, 2]
    # snn[:, 4] = sn[:, 3]
    return snn

def from_nn_state(snn: np.ndarray, norm: list) -> np.ndarray:
    #in_shape = snn.shape
    #s = np.zeros((in_shape[0], 4))
    # s[:, 0] = snn[:, 0]
    # s[:, 1] = np.arctan2(snn[:, 2], snn[:, 1])
    # s[:, 2] = snn[:, 3]
    # s[:, 3] = snn[:, 4]
    return snn * norm

def to_nn_action(u: np.ndarray, norm: float) -> np.ndarray:
    return u / norm

def from_nn_action(u: np.ndarray, norm: float) -> np.ndarray:
    return u * norm

state_symbols = ["x", "θ", "x'", "θ'"]
state_and_control_symbols = ["x", "θ", "x'", "θ'", "u"]
overall_symbols = ["goal", "state lim", "control_lim", "overall"]

def metrics_print(description:str,  symbols: List[str], vals:np.ndarray):
    line = (
            f"{description} | "
            + " | ".join(f"{s}: {v:.4f}" for s, v in zip(symbols, vals))
            + " |"
    )
    print(line)

@dataclass
class TrajMetricInd:
    # locations of interesting trajectories - for dataset browsing
    max_goal_error_ind: np.ndarray
    max_control_error_ind: int
    max_state_error_ind: np.ndarray
    median_goal_error_ind: np.ndarray

@dataclass
class TrajMetric:
    # note: all data are normalized by state and control nn normalization
    # extrema and typ. values
    max_goal_error: np.ndarray
    max_control_error: np.ndarray
    max_state_error: np.ndarray
    median_goal_error: np.ndarray

    # success rates
    goal_success: np.ndarray
    all_goal_success: float
    state_success: np.ndarray
    all_state_success: float
    control_success: np.ndarray
    all_control_success: float
    all_success: float

    # pretty print
    def pretty_print(self):
        metrics_print(
            "     max goal error (norm)",
            state_symbols,
            self.max_goal_error)
        metrics_print(
            "     median goal error (norm)",
            state_symbols,
            self.median_goal_error)
        metrics_print(
            "     goal success rate",
            state_symbols,
            self.goal_success)
        metrics_print(
            "     constraint max error (norm)",
            state_and_control_symbols,
            np.hstack((self.max_state_error, self.max_control_error)))
        metrics_print(
            "     constraint success rate",
            state_and_control_symbols,
            np.hstack((self.state_success, self.control_success)))
        metrics_print(
            "     overall success rate",
            overall_symbols,
            np.array([self.all_goal_success, self.all_state_success, self.all_control_success, self.all_success]))

class TrajectoryMetrics:
    def __init__(self):
        # use limits from the trajectory solver for normalization
        self.state_normalization = [0.22, np.pi, 0.8, 5 * np.pi]
        self.action_normalization = [0.8, ]

        self.env_params = CartpoleEnvironmentParams(pole_length=0.395,
                                               pole_mass=0.087,
                                               cart_mass=0.230,
                                               cart_length=0.044,
                                               track_length=0.44,
                                               max_cart_force=1.77,
                                               max_cart_speed=0.8,
                                               cart_tau=0.25,
                                               n=4,
                                               m=1)

        self.solver_params = SolverParams(dt=0.05,
                                     P=1e3 * np.eye(4),
                                     Q=np.diag([10, 2, 1, 0.25]),
                                     # Q = np.diag([1e-2, 1.0, 1e-3, 1e-3]) (quadratic cost)
                                     R=0.001 * np.eye(1),
                                     rho=0.05,
                                     eps=0.005,
                                     max_iters=1000,
                                     u_max=np.array([0.8]),
                                     s_max=np.array([0.44 / 2.0, 1000, 0.8, 5 * np.pi])[None, :])

        self.sc = TrajectoryScenario(s_goal=np.array([0.0, np.pi, 0.0, 0.0]),
                                s0=np.array([0.0, 0.0, 0.0, 0.0]),
                                t0=0.0,
                                T=3.5)

        self.solver = CartpoleSolverVelocity(sp=self.solver_params, ep=self.env_params)

        self.rand_state_vec = []
        self.batch_size = -1

        self.t = np.arange(self.sc.t0, self.sc.T + self.solver_params.dt, self.solver_params.dt)
        self.N = self.t.size - 1

        # give goals some slack - 10%, but perhaps a bit more on x
        self.goal_limits = np.array(self.state_normalization) * np.array([0.25, 0.1, 0.1, 0.1])
        # give this limits 5% slack
        self.control_limits = self.solver_params.u_max * np.array([0.05])
        self.state_limits = np.array(self.state_normalization) * np.array([0.05, 0.05, 0.05, 0.05])

    def reset_accumulators(self):
        """reset metric data accumulators before processing a batch."""
        self.goal_error_list = []
        self.max_traj_control_error_list = []
        self.max_traj_state_error_list = []

    def reset_random_batches(self, state_vec: np.ndarray, num_batches: int, batch_size: int, seed: int):
        """ run this once at the beginning of usage."""
        np.random.seed(seed)
        # generate s0 batches
        self.rand_state_vec = state_vec
        self.rand_batches=[]
        self.batch_size = batch_size
        self.num_batches = num_batches
        for i_batch in range(0, num_batches):
            s0_batch = np.squeeze(np.random.uniform(-state_vec, state_vec, size=(self.batch_size, 4)))
            self.rand_batches.append(s0_batch)

    def eval_metrics(self, model: nn.Sequential, loader: DataLoader, device):
        """ evaluate metrics across all batches provided by the data loader.
        I am assuming the loader provides initial states by batch."""

        for s0_batch in loader:
            # these are pytorch tensors which can be fed to the model
            # note I am assuming this batch is nn-friendly - or normalized data
            s0 = s0_batch[0]
            self.collect_batch_data(model, s0)

        # reduce data across all batches.
        metrics, _ = self.collect_metrics()
        return metrics

    def collect_random_batch(self, model: nn.Sequential, i_batch: int, return_traj: bool = False) -> None | Tuple[np.ndarray, np.ndarray]:
        s0_batch = self.rand_batches[i_batch]
        s0_nn = to_nn_state(s0_batch, self.state_normalization)
        s_nn = torch.tensor(s0_nn)
        # these are pytorch tensors which can be fed to the model
        return self.collect_batch_data(model, s_nn, return_traj=return_traj)

    def eval_metrics_rand(self, model: nn.Sequential, device):

        for i_batch in range(0, self.num_batches):
            self.collect_random_batch(model, i_batch)

        # reduce data across all batches.
        metrics, _ = self.collect_metrics()
        return metrics

    def collect_metrics(self) -> Tuple[TrajMetric, TrajMetricInd]:
        """collect metrics data across batches."""

        # collect
        goal_error = np.concatenate(self.goal_error_list, axis=0)
        max_traj_control_error = np.concatenate(self.max_traj_control_error_list, axis=0)
        max_traj_state_error = np.concatenate(self.max_traj_state_error_list, axis=0)

        #compute
        # worst-case errors
        # goal [4,]
        max_goal_error = np.max(goal_error, axis=0)
        # useful for looking at cases
        max_goal_error_ind = np.argmax(goal_error, axis=0)
        median_goal_error = np.median(goal_error, axis=0)
        median_goal_error_ind = np.argmax(goal_error==median_goal_error, axis=0)

        # state constraint [4,]
        # max along trajectories [batch_size, 4]
        max_state_error = np.max(max_traj_state_error,axis=0)
        # useful for looking at cases
        max_state_error_ind = np.argmax(max_traj_state_error,axis=0)

        # control constraint [1,]
        max_control_error = np.max(max_traj_control_error ,axis=0)
        max_control_error_ind = np.argmax(max_traj_control_error ,axis=0)
        #median_control_error = np.median(max_traj_control_error, axis=0)

        # compute accuracies - i.e., fraction of cases within these limits,
        # both separately and together

        # success logicals
        # [batch_size, 4]
        goal_success = goal_error < self.goal_limits
        state_success = max_traj_state_error < self.state_limits
        # [batch_size, 1]
        control_success = max_traj_control_error < self.control_limits

        # combined logicals
        all_goal_success = np.all(goal_success, axis=1)
        all_state_success = np.all(state_success, axis=1)
        all_control_success = np.all(control_success, axis=1)
        all_success = all_goal_success & all_state_success & all_control_success

        tm = TrajMetric(
            max_goal_error=max_goal_error / self.state_normalization,
            max_control_error=max_control_error / self.action_normalization,
            max_state_error = max_state_error / self.state_normalization,
            median_goal_error = median_goal_error / self.state_normalization,
            goal_success = np.mean(goal_success, axis=0),
            all_goal_success = np.mean(all_goal_success, axis=0),
            state_success = np.mean(state_success, axis=0),
            all_state_success = np.mean(all_state_success, axis=0),
            control_success = np.mean(control_success, axis=0),
            all_control_success = np.mean(all_control_success, axis=0),
            all_success = np.mean(all_success, axis=0)
        )

        tm_ind = TrajMetricInd(
            max_goal_error_ind=max_goal_error_ind,
            max_state_error_ind=max_state_error_ind,
            max_control_error_ind=max_control_error_ind,
            median_goal_error_ind=median_goal_error_ind
            )

        return tm, tm_ind

    def collect_batch_data(self, model: nn.Sequential, s0_batch: Tensor, return_traj: bool=False) -> None | Tuple[np.ndarray, np.ndarray]:
        """roll out trajectories across the whole batch. collect stuff."""
        s_list = []
        u_list = []
        s_nn = s0_batch.clone()
        p = next(model.parameters())
        with torch.no_grad():
            for i in range(0,self.N):
                s_nn_to = s_nn.to(dtype=p.dtype, device=p.device)
                u_nn = model(s_nn_to)
                # to numpy
                s_nn_np = s_nn.detach().cpu().numpy()
                u_nn_np = u_nn.detach().cpu().numpy()
                # to non-nn state
                s_np = from_nn_state(s_nn_np, self.state_normalization)
                u_np = from_nn_action(u_nn_np, self.action_normalization)
                s_list.append(np.expand_dims(s_np, 1))
                u_list.append(np.expand_dims(u_np, 1))
                # to torch
                s = torch.tensor(s_np)
                u = torch.tensor(u_np)
                # step the integrator
                s_next = self.solver.step(s,u)
                # back to numpy for conversion
                s_next_np = s_next.detach().cpu().numpy()
                s_next_nn_np = to_nn_state(s_next_np, self.state_normalization)
                s_nn = torch.tensor(s_next_nn_np)

        # collect the last state
        s_list.append(np.expand_dims(s_next_np, 1))

        # collect numpy arrays
        all_s = np.concatenate(s_list, axis=1)
        all_u = np.concatenate(u_list, axis=1)

        # collect the final state
        last_s = all_s[:,-1,:]

        # compute raw errors
        # goal error [batch_size, 4]
        goal_error_batch = np.abs(last_s - self.sc.s_goal)

        # state constraint error - [batch_size, traj_size, 4]
        control_error_plus = np.maximum(all_s - self.solver_params.s_max, 0.0)
        control_error_minus = np.minimum(all_s + self.solver_params.s_max, 0.0)
        control_error_batch = np.maximum(control_error_plus, np.abs(control_error_minus))
        # reduce along traj dim - [batch_size, 4]
        max_traj_state_error_batch = np.max(control_error_batch, axis=1)

        # control constraint error - [batch_size, traj_size, 1]
        control_error_plus = np.maximum(all_u - self.solver_params.u_max, 0.0)
        control_error_minus = np.minimum(all_u + self.solver_params.u_max, 0.0)
        control_error_batch = np.maximum(control_error_plus, np.abs(control_error_minus))
        # reduce along traj dim - [batch_size, 1]
        max_traj_control_error_batch = np.max(control_error_batch, axis=1)

        # accumulate
        self.goal_error_list.append(goal_error_batch)
        self.max_traj_control_error_list.append(max_traj_control_error_batch)
        self.max_traj_state_error_list.append(max_traj_state_error_batch)

        # void return
        if not return_traj:
            return None
        else:
            return all_s, all_u
