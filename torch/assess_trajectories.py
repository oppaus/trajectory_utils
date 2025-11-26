
from trajectory import TrajectoryScenario, Trajectory, TrajectoryExpert
from cartpole_solver_velocity import CartpoleSolverVelocity, CartpoleEnvironmentParams, SolverParams

from plot_trajectory import plot_trajectory
import numpy as np
import pickle
import os

def main():
    tdir = "trajectories_test6"
    good_data = 0
    num_data = 0
    for file in os.listdir(tdir):
        file = os.path.join(tdir, file)
        # unpickle!
        #pickle_fest = [env_params, solver_params, traj]
        with open(file,"rb") as f:
            pickle_fest = pickle.load(f)
        traj: Trajectory = pickle_fest[2]
        num_data += 1
        if traj.conv:
            good_data += 1
        # plot!
        # plot_trajectory(env_params=env_params,
        #                 traj=traj, filename_base="cartpole_velocity_unpickle",
        #                 animate=True)
    print(good_data/num_data)

if __name__ == "__main__":
    main()