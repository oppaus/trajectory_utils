""" Load NN model and predict a closed-loop trajectory."""

import torch
import os
import torch.nn as nn
import numpy as np
from scp_solver import SolverParams
from cartpole_solver_velocity import CartpoleSolverVelocity, CartpoleEnvironmentParams
from trajectory import TrajectoryScenario, Trajectory
from plot_trajectory import plot_trajectory

def to_nn_state(s: np.ndarray, norm: list) -> np.ndarray:
    snn = np.zeros((5,))
    sn = s / norm
    snn[0] = sn[0]
    snn[1] = np.cos(sn[1])
    snn[2] = np.sin(sn[1])
    snn[3] = sn[2]
    snn[4] = sn[3]
    return snn

def from_nn_state(snn: np.ndarray, norm: list) -> np.ndarray:
    s = np.zeros((4,))
    s[0] = snn[0]
    s[1] = np.arctan2(snn[2], snn[1])
    s[2] = snn[3]
    s[3] = snn[4]
    return s * norm


def to_nn_action(u: np.ndarray, norm: float) -> np.ndarray:
    return u / norm

def from_nn_action(u: np.ndarray, norm: float) -> np.ndarray:
    return u * norm

def predict(trained_model_dir: str, trained_model_epoch: list, output_filename: str):
    # use limits from the trajectory solver for normalization
    state_normalization = [0.22, 1, 0.8, 5*np.pi]
    action_normalization = 0.8

    env_params = CartpoleEnvironmentParams(pole_length=0.395,
                                           pole_mass=0.087,
                                           cart_mass=0.230,
                                           cart_length=0.044,
                                           track_length=0.44,
                                           max_cart_force=1.77,
                                           max_cart_speed=0.8,
                                           cart_tau=0.25,
                                           n=4,
                                           m=1)

    solver_params = SolverParams(dt=0.05,
                                 P=1e3 * np.eye(4),
                                 Q=np.diag([10, 2, 1, 0.25]),  # Q = np.diag([1e-2, 1.0, 1e-3, 1e-3]) (quadratic cost)
                                 R=0.001 * np.eye(1),
                                 rho=0.05,
                                 eps=0.005,
                                 max_iters=1000,
                                 u_max=np.array([0.8]),
                                 s_max=np.array([0.44 / 2.0, 1000, 0.8, 5 * np.pi])[None, :])

    sc = TrajectoryScenario(s_goal = np.array([0.0, np.pi, 0.0, 0.0]),
                            s0 = np.array([0.0, 0.0, 0.0, 0.0]),
                            t0 = 0.0,
                            T = 3.5)

    t = np.arange(sc.t0, sc.T + solver_params.dt, solver_params.dt)
    N = t.size - 1

    solver = CartpoleSolverVelocity(sp=solver_params, ep=env_params)

    tdir = trained_model_dir
    epoch_num = trained_model_epoch
    fdir = os.path.join("train",tdir)
    epoch_str = ""
    for sub_epoch in trained_model_epoch:
        epoch_str += "_"
        epoch_str += str(sub_epoch)
    fname = tdir + epoch_str + ".pth"
    fpath = os.path.join(fdir, fname)
    ckpt = torch.load(fpath, map_location="cpu")

    input_dim = ckpt["input_dim"]
    #hidden_dim = ckpt["hidden_dim"]
    hidden_dim = 32
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

    s0_nn = to_nn_state(sc.s0, state_normalization)
    s_nn = torch.tensor(s0_nn)
    traj = Trajectory(s=np.zeros((N+1,4)),
                      u=np.zeros((N,1)),
                      N=N,
                      dt=solver_params.dt,
                      J=np.array([]),
                      energy1=np.array([]),
                      energy2=np.array([]),
                      sc=sc,
                      conv=True,
                      status="NN_prediction")

    s_list = []
    u_list = []
    for i in range(0,N):
        with torch.no_grad():
            u_nn = model(s_nn)
        # to numpy
        s_nn_np = s_nn.detach().cpu().numpy()
        u_nn_np = u_nn.detach().cpu().numpy()
        # to non-nn state
        s_np = from_nn_state(s_nn_np, state_normalization)
        u_np = from_nn_action(u_nn_np, action_normalization)
        s_list.append(s_np)
        u_list.append(u_np)
        # to torch
        s = torch.tensor(s_np)
        u = torch.tensor(u_np)
        # step the integrator
        s_next = solver.step(s,u)
        # back to numpy for conversion
        s_next_np = s_next.detach().cpu().numpy()
        s_next_nn_np = to_nn_state(s_next_np, state_normalization)
        s_nn = torch.tensor(s_next_nn_np)

    # collect the last state
    s_list.append(s_next_np)

    # collect numpy arrays
    traj.s = np.vstack(s_list)
    traj.u = np.vstack(u_list)

    # compute energies
    traj.energy1 = solver.energy.compute_energy(traj.s)
    traj.energy2 = solver.energy.compute_energy_pole(traj.s)

    # off to the plotter
    plot_trajectory(solver_params=solver_params,
                    env_params=env_params,
                    traj=traj,
                    filename_base=output_filename,
                    animate=True)


if __name__ == "__main__":
    predict(trained_model_dir="trajectories_big_1",
            trained_model_epoch=[0,375],
            output_filename="nn_test_2")
