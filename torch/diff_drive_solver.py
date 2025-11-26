"""
Differential drive robot control trajectory planner.
"""
from scp_solver import SCPSolver, SolverParams

import cvxpy as cvx
import torch
import numpy as np
from typing import Tuple, Callable, Any, List

from scalar_field_interpolator import ScalarFieldInterpolator, SDF

class DiffDriveSolver(SCPSolver):
    sdf_interpolator: ScalarFieldInterpolator
    u_goal: np.ndarray
    u_min: np.ndarray
    u_final: np.ndarray
    rho_u: float
    # linearization variables
    A_param_sdf: List[cvx.Parameter]
    B_param_sdf: List[cvx.Parameter]
    c_param_sdf: List[cvx.Parameter]
    slack_obs: cvx.Variable

    def __init__(self, sp:SolverParams, u_min, sdf: SDF, rho_u):
        super().__init__(sp=sp)
        self.u_min = u_min
        self.rho_u = rho_u
        self.setup_this(sdf)

    def setup_this(self, sdf: SDF):
        super().setup()
        # add the function for the sdf interpolator
        self.sdf_interpolator = ScalarFieldInterpolator(sdf.sdf, sdf.ox, sdf.oy, sdf.res)

    def reset_custom(self, s0:np.ndarray, u_goal:np.ndarray, u_final:np.ndarray, N:int):
        """
        N : int
            The time horizon (N * dt) of the solver.
        u_goal : numpy.ndarray
            The goal controls.
        u_final : numpy.ndarray
            The target control at the final state.
        s0 : numpy.ndarray
            The initial state.
        """
        # prepare for a new solve
        self.s0 = s0
        self.s_goal = np.array([])
        self.N = N
        self.s0 = s0
        self.u_final = u_final
        self.u_goal = u_goal
        n = self.params.Q.shape[0]
        m = self.params.R.shape[0]
        # declare additional optimization parameters and variables
        self.A_param_sdf = [cvx.Parameter((1, n)) for _ in range(self.N+1)]
        self.B_param_sdf = [cvx.Parameter((1, m)) for _ in range(self.N)]
        self.c_param_sdf = [cvx.Parameter(1) for _ in range(self.N+1)]
        self.slack_obs = cvx.Variable(self.N, nonneg=True)  # one slack per time
        self.reset_core()

    def opt_problem(self) -> cvx.Problem:
        # set up cvxpy optimization
        objective = self.opt_problem_objective(self.s_cvx, self.u_cvx)
        #objective += cvx.sum(self.slack_obs)
        constraints = [self.s_cvx[i + 1] == self.c_param[i] + self.A_param[i] @ self.s_cvx[i] +
                       self.B_param[i] @ self.u_cvx[i] for i in range(self.N)]
        # note this does not enforce initial robot heading, only position
        constraints += [self.s_cvx[0,:2] == self.s0[:2]]
        constraints += [self.u_cvx[self.N-1] == self.u_final]
        constraints += [cvx.abs(self.u_cvx) <= self.params.u_max]
        constraints += [self.u_cvx >= self.u_min]
        # obstacle avoidance
        #constraints += [self.c_param_sdf[i] + self.A_param_sdf[i] @ self.s_cvx[i] + self.slack_obs[i] >= 0.0 for i in range(1, self.N)]
        constraints += [self.c_param_sdf[i] + self.A_param_sdf[i] @ self.s_cvx[i] >= 0.0 for i in range(1, self.N)]
        constraints += [cvx.max(cvx.abs(self.s_cvx - self.s_prev_param)) <= self.params.rho]
        constraints += [cvx.max(cvx.abs(self.u_cvx - self.u_prev_param)) <= self.rho_u]

        prob = cvx.Problem(cvx.Minimize(objective), constraints)
        return prob

    def opt_problem_objective(self, s: cvx.Expression, u: cvx.Expression) -> cvx.Expression:
        """ The objective function of the problem to solve. Used for adaptive trust region. In cvxpy-speak."""
        objective = cvx.sum([cvx.quad_form(self.u_cvx[i] - self.u_goal, self.params.R) for i in range(self.N)])
        return objective

    def ode(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        s: [..., 3]  (x, y, θ)
        u: [..., 2]  (v, omega)
        returns ds/dt with shape [..., 4]
        """

        x, y, th = s[..., 0], s[..., 1], s[..., 2]
        v, omega = u[..., 0], u[..., 1]
        sin_th, cos_th = torch.sin(th), torch.cos(th)

        dx = v * cos_th
        dy = v * sin_th
        dth = omega

        ds = torch.stack((dx, dy, dth), dim=-1)
        return ds

    def linearize_constraints(self, s: np.ndarray, u: np.ndarray):
        A, B, c = self.linearize(self.sdf_interpolator.interpolator, s[:-1], u)
        for i2 in range(self.N):
            self.A_param_sdf[i2].value = A[i2]
            self.B_param_sdf[i2].value = B[i2]
            self.c_param_sdf[i2].value = c[i2]

    def initialize_trajectory(self):
        super().initialize_trajectory()
        # not a good idea. this will generally be infeasible - and it is not
        # easy for cvxpy to rescue itself from trajectories that go
        # off the map!
        # for i in range(self.u_init.shape[0]):
        #     self.u_init[i,:] = self.u_goal
