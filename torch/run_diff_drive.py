"""
Differential drive robot control trajectory planner.
"""

from diff_drive_solver import DiffDriveSolver, SolverParams
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.func import vmap

from scalar_field_interpolator import SDF

def main():
    # Define occupancy map and obstacle avoidance signed distance function
    robot_radius = 0.14
    obstacle_radius = 0.14
    max_robot_v = 0.3
    max_robot_omega = 1.0
    x_size = 3.0
    y_size = 2.0
    x_origin = -1.5
    y_origin = -1.0
    resolution = 0.02
    sdf = SDF(x_size, y_size, x_origin, y_origin, resolution)
    sdf.generate(robot_radius, obstacle_radius)
    # state dim
    n = 3
    #control dim
    m = 2
    # start state: robot centered in the room, not moving
    s0 = np.array([0.0, 0.0, 0.0])
    # control goal: the real target
    u_goal = np.array([0.3, 0.3])
    # control final: not moving
    u_final = np.array([0.0, 0.0])
    T = 30.0

    rho_u = 0.02 # trust region size - control
    u_min = np.array([0.0, -100.0])[None,:]
    dt = 0.1
    solver_params = SolverParams(dt=dt,
                              P=1.0 * np.eye(n),
                              Q=np.eye(n),
                              R=np.eye(m),
                              rho=0.05,
                              eps=0.001,
                              max_iters=10000,
                              u_max=np.array([max_robot_v, max_robot_omega])[None,:],
                              s_max=np.array([]))

    t = np.arange(0.0, T + dt, dt)
    N = t.size - 1

    solver = DiffDriveSolver(sp=solver_params, u_min=u_min, sdf=sdf, rho_u=rho_u)
    solver.reset_custom(s0, u_goal, u_final, N)
    solver.initialize_trajectory()

    s, u, J, conv, status = solver.solve()

    print("SCP convergence: " + str(conv))

    # open-loop rollout
    s, u = solver.rollout(s, u)

    # Plot state and control trajectories
    fig, ax = plt.subplots(2, 3, dpi=150, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.45)
    labels_s = (r"$x(t)$", r"$y(t)$", r"$\theta(t)$", r"$\dot{x}(t)$", r"$\dot{y}(t)$", r"$\dot{\theta}(t)$")
    labels_u = (r"$v(t)$", r"$\omega(t)$")
    for i in range(n):
        row = i // 3
        col = i % 3
        ax[row, col].plot(t, s[:, i])
        #ax[row, col].axhline(s_goal[i], linestyle="--", color="tab:orange")
        ax[row, col].set_xlabel(r"$t$")
        ax[row, col].set_ylabel(labels_s[i])
    for i in range(m):
        ax[1, i].plot(t[:-1], u[:, i])
        ax[1, i].axhline(solver_params.u_max[:,i], linestyle="--", color="tab:orange")
        ax[1, i].axhline(-solver_params.u_max[:,i], linestyle="--", color="tab:orange")
        ax[1, i].set_xlabel(r"$t$")
        ax[1, i].set_ylabel(labels_u[i])
    ax[1, 2].plot(s[:,0], s[:, 1])
    ax[1, 2].set_xlabel(labels_s[0])
    ax[1, 2].set_ylabel(labels_s[1])
    ax[1, 2].set_aspect('equal')
    plt.savefig("diff_drive_state.png", bbox_inches="tight")
    plt.show()

    # plot trajectory over obstacle constraints
    dx = 0.05
    x = np.arange(sdf.ox, sdf.ox + sdf.x_size + dx, dx)
    y = np.arange(sdf.oy, sdf.oy + sdf.y_size + dx, dx)
    xx, yy = np.meshgrid(x, y)

    s_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    u_pts = np.zeros(s_pts.shape)
    S = torch.from_numpy(s_pts)
    U = torch.from_numpy(u_pts)

    c = vmap(solver.sdf_interpolator.interpolator, in_dims=(0, 0))(S, U)  # (T,)
    c_np = c.detach().cpu().numpy()
    c_np = np.reshape(c_np, (41, 61))
    plt.figure()
    plt.imshow(c_np,
               origin='lower',
               extent=[x.min(), x.max(), y.min(), y.max()],  # map array to coordinate bounds
               aspect='equal',  # or 'equal'
               cmap='viridis'
               )
    plt.colorbar()
    plt.title('trajectory and S.D.F.')
    plt.plot(s[:,0], s[:,1], linestyle="-", color="tab:red")
    plt.xlabel(labels_s[0])
    plt.ylabel(labels_s[1])
    plt.savefig("diff_drive_traj.png", bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(1, 1, dpi=150, figsize=(8, 5))
    ax.semilogy(J)
    ax.set_xlabel(r"SCP iteration $i$")
    ax.set_ylabel(r"SCP cost $J(\bar{x}^{(i)}, \bar{u}^{(i)})$")
    plt.savefig("diff_drive_cost.png", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()