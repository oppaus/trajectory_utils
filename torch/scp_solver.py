"""
This code generalizes/optimizes sequential convex programming (SCP) as espoused by the
Autonomous Systems Lab (ASL), Stanford University (e.g., AA203). CVXPY is the solver
and pytorch is utilized for in-process compatibility with other machine learning needs.
This is most certainly not the fastest way to get a solution, but you don't have to
compute a jacobian and the optimization problem definition is very clean. Particular
solvers are subclasses of this class - which implement the virtual methods for objectives
and constraints. A useful addition would be an adaptive trust region. The adaptive
trust region details depend on the problem being solved.
"""
from copy import deepcopy

import cvxpy as cvx
import torch
from torch.func import vmap, jacrev
import numpy as np
from typing import Tuple, Callable, Any, List
import time
from dataclasses import dataclass

torch.set_default_dtype(torch.float64)

@dataclass
class SolverParams:
    """the parameters of an SCP (solver) expert"""
    dt: float
    # terminal state cost
    P: np.ndarray
    # state cost along trajectory
    Q: np.ndarray
    # control cost matrix
    R: np.ndarray
    # trust region size
    rho: float
    # convergence tolerance
    eps: float
    # maximum number of SCP iterations
    max_iters: int
    # state max
    s_max: np.ndarray
    # control max
    u_max: np.ndarray

class SCPSolver:
    # basic SCP parameters (see the init method for docstrings)
    N : int
    params: SolverParams
    s_goal: np.ndarray
    s0: np.ndarray
    # initial trajectory
    s_init: np.ndarray
    u_init: np.ndarray
    # cvxpy stuff (for optimizing CVXPY runtime):
    # the cvxpy optimization problem
    prob: cvx.Problem
    # state and control variables
    s_cvx: cvx.Variable
    u_cvx: cvx.Variable
    # linearization variables
    A_param: List[cvx.Parameter]
    B_param: List[cvx.Parameter]
    c_param: List[cvx.Parameter]
    # previous stats and control parameters
    s_prev_param: cvx.Parameter
    u_prev_param: cvx.Parameter
    # the discretized, pytorch-ified dynamics
    fd: Callable

    def __init__(self,
                 sp: SolverParams):
        """ Arguments
        ---------
        sp : SolverParams
            Time interval between trajectory points.
        """
        self.params = sp

    def setup(self):
        # these virtual-method calls do not belong in the constructor!
        #self.prob = self.opt_problem()
        self.fd = self.build_fd_ode(self.params.dt)

    def reset(self, s0:np.ndarray, s_goal:np.ndarray, N:int):
        """
        N : int
            The time horizon (N * dt) of the solver.
        s_goal : numpy.ndarray
            The goal state.
        s0 : numpy.ndarray
            The initial state.
        """
        # prepare for a new solve
        self.s0 = s0
        self.s_goal = s_goal
        self.N = N
        self.reset_core()

    def reset_core(self):
        """
            Run the core reset of the optimization problem.
            Be sure all your class variables are set up.
        """
        n = self.params.Q.shape[0]
        m = self.params.R.shape[0]
        # declare all optimization parameters and variables
        # this speeds up CVXPY solves by a factor of 5 or more.
        self.s_cvx = cvx.Variable((self.N + 1, n))
        self.u_cvx = cvx.Variable((self.N, m))
        self.A_param = [cvx.Parameter((n, n)) for _ in range(self.N)]
        self.B_param = [cvx.Parameter((n, m)) for _ in range(self.N)]
        self.c_param = [cvx.Parameter(n) for _ in range(self.N)]
        self.s_prev_param = cvx.Parameter((self.N + 1, n))
        self.u_prev_param = cvx.Parameter((self.N, m))
        self.prob = self.opt_problem()

    def linearize(self, f: Callable, s_np: np.ndarray, u_np: np.ndarray) ->\
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        f: torch function (s,u) -> y  (supports batching on leading dim)
        s_np: [n] or [T,n]  numpy
        u_np: [m] or [T,m]  numpy

        Returns (A, B, c) as numpy:
          if unbatched: A[n,n], B[n,m], c[n]
          if batched:   A[T,n,n], B[T,n,m], c[T,n]
        """
        # to torch (double)
        S = torch.as_tensor(s_np, dtype=torch.float64)
        U = torch.as_tensor(u_np, dtype=torch.float64)

        jac_fun = jacrev(f, argnums=(0, 1))  # returns (df/ds, df/du)

        if S.ndim == 1:  # unbatched
            c = f(S, U)                      # [n]
            A, B = jac_fun(S, U)             # [n,n], [n,m]
            d = c - A @ S - B @ U
        else:            # batched over leading T
            # vectorize both evaluation and jacobians over leading axis
            c  = vmap(f, in_dims=(0, 0))(S, U)                # [T,n]
            A, B = vmap(jac_fun, in_dims=(0, 0))(S, U)
            # This rolls the correction to the current linearization point into the c
            # value. You could correct in the optimization specification, EXCEPT that this
            # breaks "Disciplined Convex Programming" due to use of cvxpy parameters in this
            # solver. That einsum stuff is -A@S - B@U.
            d = c - torch.einsum('tij,tj->ti', A, S) - torch.einsum('tij,tj->ti', B, U)
        # back to numpy
        d_ret = d.detach().cpu().numpy()
        A_ret = A.detach().cpu().numpy()
        B_ret = B.detach().cpu().numpy()
        return A_ret, B_ret, d_ret

    def linearize_dynamics(self, s: np.ndarray, u: np.ndarray):
        A, B, c = self.linearize(self.fd, s[:-1], u)
        for i2 in range(self.N):
            self.A_param[i2].value = A[i2]
            self.B_param[i2].value = B[i2]
            self.c_param[i2].value = c[i2]

    def linearize_constraints(self, s: np.ndarray, u: np.ndarray):
        pass

    def initialize_trajectory(self):
        """Set and rollout the zero control trajectory."""
        n = self.params.Q.shape[0]  # state dimension
        m = self.params.R.shape[0]  # control dimension
        # the zero control trajectory
        self.u_init = np.zeros((self.N, m))
        self.s_init = np.zeros((self.N + 1, n))
        self.s_init[0] = self.s0
        self.s_init, self.u_init = self.rollout(self.s_init, self.u_init)

    def set_trajectory(self, s_init, u_init):
        """simply set the trajectory."""
        self.s_init = s_init
        self.u_init = u_init

    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, str]:
        """Solve over the time horizon via SCP.

        Arguments from class
        ---------
        eps : float
            Objective value for SCP convergence.
        max_iters : int
            Maximum number of SCP iterations.

        Returns
        -------
        s : numpy.ndarray
            s[k] is the state at time step k
        u : numpy.ndarray
            u[k] is the control at time step k
        J : numpy.ndarray
            J[i] is the SCP cost after the i-th solver iteration
        conv: bool
            convergence state
        """

        # Initialize trajectory
        s = deepcopy(self.s_init)
        u = deepcopy(self.u_init)

        # SCP solve loop
        converged = False
        J = np.zeros(self.params.max_iters + 1)
        J[0] = np.inf
        start_time = time.time()
        for i in range(self.params.max_iters):
            self.linearize_dynamics(s, u)
            self.linearize_constraints(s, u)
            self.s_prev_param.value = s
            self.u_prev_param.value = u
            self.prob.solve(solver=cvx.SCS, warm_start=True, eps=1e-3, max_iters=40000)
            if self.prob.status != "optimal":
                print("SCP solve failed. CVXPY problem status: " + self.prob.status)
                break
            s = self.s_cvx.value
            u = self.u_cvx.value
            J[i+1] = self.prob.objective.value
            dJ = np.abs(J[i + 1] - J[i])
            if dJ < self.params.eps:
                converged = True
                print("SCP converged after {} iterations.".format(i))
                break
        J = J[1 : i + 1]
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Solve time: {elapsed_time:.4f} seconds")
        return s, u, J, converged, self.prob.status

    def discretize(self, f:Callable, dt: float) -> Callable:
        """
        f: (s, u) -> ds/dt   (both torch tensors; supports batching)
        returns fd(s,u) that maps to next state with Runge-Kutta 4th order integration.
            That is, s function describing the discrete-time dynamics, such that
            `s[k+1] = fd(s[k], u[k])`.
        See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods .
        """
        def integrator(s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            k1 = dt * f(s, u)
            k2 = dt * f(s + 0.5 * k1, u)
            k3 = dt * f(s + 0.5 * k2, u)
            k4 = dt * f(s + k3, u)
            return s + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        return integrator

    def build_fd_ode(self, dt: float) -> Callable:
        return self.discretize(self.ode, dt)

    def rollout(self, s: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        S = torch.as_tensor(s, dtype=torch.float64)
        U = torch.as_tensor(u, dtype=torch.float64)
        for k in range(self.N):
            S[k + 1] = self.fd(S[k], U[k])
        s = S.detach().cpu().numpy()
        u = U.detach().cpu().numpy()
        return s, u

    def ode(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        returns
            A tensor ds/dt describing the continuous-time dynamics: `ds/dt = f(s, u)`.
        """
        raise NotImplementedError()

    def opt_problem(self) -> cvx.Problem:
        """ The optimization problem to solve - in cvxpy-speak."""
        raise NotImplementedError()

    def opt_problem_objective(self, s: cvx.Expression, u: cvx.Expression) -> cvx.Expression:
        """ The objective function of the problem to solve. Used for adaptive trust region. In cvxpy-speak."""
        raise NotImplementedError()


