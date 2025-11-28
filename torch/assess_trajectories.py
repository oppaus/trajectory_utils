
from trajectory import TrajectoryScenario, Trajectory, TrajectoryExpert
from cartpole_solver_velocity import CartpoleSolverVelocity, CartpoleEnvironmentParams, SolverParams

from plot_trajectory import plot_trajectory
import numpy as np
import pickle
import os

def main(tdir: str):
    good_data = 0
    num_data = 0
    for file in os.listdir(tdir):
        file = os.path.join(tdir, file)
        # unpickle!
        #pickle_fest = [env_params, solver_params, traj]
        with open(file,"rb") as f:
            pickle_fest = pickle.load(f)
        traj: Trajectory = pickle_fest[2]
        sp: SolverParams = pickle_fest[1]
        ep: CartpoleEnvironmentParams = pickle_fest[0]
        num_data += 1
        if traj.conv:
            good_data += 1
            # plot!
            if (num_data % 10000) == 0:
                plot_trajectory(solver_params=sp, env_params=ep,
                                 traj=traj, filename_base="",
                                 animate=False)
    print(good_data, num_data, good_data/num_data)

if __name__ == "__main__":
    main("trajectories_big_1")