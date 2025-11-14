"""
Cart velocity (servo) controlled cartpole.
"""

from cartpole_solver_velocity import CartpoleSolverVelocity

from animate_cartpole import animate_cartpole
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Define constants
    pole_length = 0.395
    pole_mass = 0.087
    cart_mass = 0.230
    cart_length = 0.044
    track_length = 0.44
    max_cart_force = 1.77
    max_cart_speed = 0.8
    # tracking time constant of the cart speed controller
    cart_tau = 0.25
    # state dim
    n = 4
    #control dim
    m = 1
    # goal state: pole upright
    s_goal = np.array([0, np.pi, 0, 0])
    # start state: pole down + noise
    s0 = np.array([0.0, 0.0, 0.0, 0.0])
    s0 = s0 + np.array([0.0, 0.2, 0.0, -0.2])
    dt = 0.05
    T = 4.5
    # terminal state cost
    P = 1e3 * np.eye(n)
    # state cost
    Q = np.diag([1e-2, 1.0, 1e-3, 1e-3])
    # control cost matrix
    R = 0.001 * np.eye(m)
    # trust region size
    rho = 0.05
    # control effort bound
    u_max = np.array([max_cart_speed])
    s_max = np.array([track_length/2.0, 1000, max_cart_speed, 1000])[None, :]
    eps = 0.005  # convergence tolerance
    #max_iters = 100  # maximum number of SCP iterations
    max_iters = 1000  # maximum number of SCP iterations
    animate = True  # flag for animation

    t = np.arange(0.0, T + dt, dt)
    N = t.size - 1

    # load the reference solution
    ref_soln = np.load("cartpole_vctrl.npz")
    s_in = ref_soln['s']
    u_in = ref_soln['u']
    last_state = s_in[-1,:]
    s_n = s_in.shape[0]
    s_m = s_in.shape[1]
    u_n = u_in.shape[0]
    u_m = u_in.shape[1]
    s_init = np.zeros((N+1, s_m))
    s_init[0,:] = s0
    u_init = np.zeros((N, u_m))
    # s_init[0:s_n,:] = s_in
    # s_init[s_n:-1,:] = last_state
    u_init[0:u_n,:] = u_in

    solver = CartpoleSolverVelocity(N, dt, P, Q, R, u_max, rho, s_goal, s0, s_max, cart_mass, pole_length, pole_mass, cart_tau)
    s,u = solver.rollout(s_init,u_init)
    solver.set_trajectory(s, u)
    s, u, J, conv = solver.solve(eps, max_iters)

    print("SCP convergence: " + str(conv))

    # rollout solution given the controls returned by SCP
    s, u = solver.rollout(s, u)

    # Plot state and control trajectories
    fig, ax = plt.subplots(2, n, dpi=150, figsize=(11, 6))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    labels_s = (r"$x(t)$", r"$\theta(t)$", r"$\dot{x}(t)$", r"$\dot{\theta}(t)$")
    labels_u = (r"$u(t)$",)
    for i in range(n):
        ax[0,i].plot(t, s[:, i])
        ax[0,i].axhline(s_goal[i], linestyle="--", color="tab:orange")
        if s_max[0,i] < 10:
            ax[0,i].axhline(s_max[0,i], linestyle="--", color="tab:orange")
            ax[0,i].axhline(-s_max[0,i], linestyle="--", color="tab:orange")
        ax[0,i].set_xlabel(r"$t$")
        ax[0,i].set_ylabel(labels_s[i])
    for i in range(m):
        ax[1,i].plot(t[:-1], u[:, i])
        ax[1,i].axhline(u_max, linestyle="--", color="tab:orange")
        ax[1,i].axhline(-u_max, linestyle="--", color="tab:orange")
        ax[1,i].set_xlabel(r"$t$")
        ax[1,i].set_ylabel(labels_u[i])
    # add the trajectory plots
    xvec = s[:, 0] + np.sin(s[:, 1])*pole_length
    yvec = - np.cos(s[:, 1])*pole_length
    ax[1, 1].plot(xvec, yvec)
    ax[1, 1].set_xlabel("x")
    ax[1, 1].set_ylabel("y")
    fig.delaxes(ax[1, 2])
    fig.delaxes(ax[1, 3])
    plt.savefig("cartpole_vctrl_state.restart.png", bbox_inches="tight")
    plt.show()

    # Plot cost history over SCP iterations
    fig, ax = plt.subplots(1, 1, dpi=150, figsize=(8, 5))
    ax.semilogy(J)
    ax.set_xlabel(r"SCP iteration $i$")
    ax.set_ylabel(r"SCP cost $J(\bar{x}^{(i)}, \bar{u}^{(i)})$")
    plt.savefig("cartpole_vctrl_cost.restart.png", bbox_inches="tight")
    plt.show()

    # Animate the solution
    if animate:
        fig, ani = animate_cartpole(t, s[:, 0], s[:, 1], pole_length, cart_length)
        ani.save("cartpole_vctrl.restart.gif", writer="ffmpeg")
        #plt.show()

if __name__ == "__main__":
    main()