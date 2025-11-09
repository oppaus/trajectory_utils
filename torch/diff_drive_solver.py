"""
Differential drive robot control trajectory planner.
"""
from scp_solver import SCPSolver

import cvxpy as cvx
import torch
import numpy as np

from scalar_field_interpolator import ScalarFieldInterpolator, SDF

class DiffDriveSolver(SCPSolver):
    sdf_interpolator: ScalarFieldInterpolator
    u_goal: np.ndarray
    u_min: np.ndarray
    u_final: np.ndarray
    rho_u: float
    def __init__(self, N, dt, P, Q, R, u_max, rho, s_goal, s0, u_goal, u_min, sdf: SDF, u_final, rho_u):
        super().__init__(N, dt, P, Q, R, u_max, rho, s_goal, s0)
        self.u_goal = u_goal
        self.u_min = u_min
        self.u_final = u_final
        self.rho_u = rho_u
        n = Q.shape[0]
        m = R.shape[0]
        self.A_param_sdf = [cvx.Parameter((1, n)) for _ in range(N+1)]
        self.B_param_sdf = [cvx.Parameter((1, m)) for _ in range(N)]
        self.c_param_sdf = [cvx.Parameter(1) for _ in range(N+1)]
        self.slack_obs = cvx.Variable(self.N, nonneg=True)  # one slack per time
        self.setup_this(sdf)

    def setup_this(self, sdf: SDF):
        super().setup()
        # add the function for the sdf interpolator
        self.sdf_interpolator = ScalarFieldInterpolator(sdf.sdf, sdf.ox, sdf.oy, sdf.res)

    def opt_problem(self) -> cvx.Problem:
        # set up cvxpy optimization
        objective = self.opt_problem_objective(self.s_cvx, self.u_cvx)
        #objective += cvx.sum(self.slack_obs)
        constraints = [self.s_cvx[i + 1] == self.c_param[i] + self.A_param[i] @ self.s_cvx[i] +
                       self.B_param[i] @ self.u_cvx[i] for i in range(self.N)]
        constraints += [self.s_cvx[0,:2] == self.s0[:2]]
        constraints += [self.u_cvx[self.N-1] == self.u_final]
        constraints += [cvx.abs(self.u_cvx) <= self.u_max]
        constraints += [self.u_cvx >= self.u_min]
        # obstacle avoidance
        #constraints += [self.c_param_sdf[i] + self.A_param_sdf[i] @ self.s_cvx[i] + self.slack_obs[i] >= 0.0 for i in range(1, self.N)]
        constraints += [self.c_param_sdf[i] + self.A_param_sdf[i] @ self.s_cvx[i] >= 0.0 for i in range(1, self.N)]
        constraints += [cvx.max(cvx.abs(self.s_cvx - self.s_prev_param)) <= self.rho]
        constraints += [cvx.max(cvx.abs(self.u_cvx - self.u_prev_param)) <= self.rho_u]

        prob = cvx.Problem(cvx.Minimize(objective), constraints)
        return prob

    def opt_problem_objective(self, s: cvx.Expression, u: cvx.Expression) -> cvx.Expression:
        """ The objective function of the problem to solve. Used for adaptive trust region. In cvxpy-speak."""
        objective = cvx.sum([cvx.quad_form(self.u_cvx[i] - self.u_goal, self.R) for i in range(self.N)])
        return objective

    def ode(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        s: [..., 6]  (x, y, θ, dx, dy)
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

    def initialize_trajectory(self, n: int, m: int):
        super().initialize_trajectory(n, m)
        # not a good idea. this will generally be infeasible - and it is not
        # necessarily easy for cvxpy to rescue itself from trajectories inside
        # of obstacles - or off the map!
        # for i in range(self.u_init.shape[0]):
        #     self.u_init[i,:] = self.u_goal
