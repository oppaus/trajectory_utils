"""
Cart velocity (servo) controlled cartpole.
"""

from scp_solver import SCPSolver

import cvxpy as cvx
import torch

class CartpoleSolverVelocity(SCPSolver):
    def __init__(self, N, dt, P, Q, R, u_max, rho, s_goal, s0, s_max, cart_mass, pole_length, pole_mass, cart_tau):
        super().__init__(N, dt, P, Q, R, u_max, rho, s_goal, s0)
        self.cart_mass = cart_mass
        self.pole_length = pole_length
        self.pole_mass = pole_mass
        self.s_max = s_max
        self.tau = cart_tau
        self.setup()

    def opt_problem(self) -> cvx.Problem:
        # set up cvxpy optimization
        objective = cvx.quad_form((self.s_cvx[self.N] - self.s_goal), self.P) + cvx.sum(
            [cvx.quad_form(self.s_cvx[i] - self.s_goal, self.Q) + cvx.quad_form(self.u_cvx[i], self.R) for i in range(self.N)])
        constraints = [self.s_cvx[i + 1] == self.c_param[i] + self.A_param[i] @ self.s_cvx[i] +
                       self.B_param[i] @ self.u_cvx[i] for i in range(self.N)]
        constraints += [self.s_cvx[0] == self.s0]
        constraints += [cvx.abs(self.u_cvx) <= self.u_max]
        constraints += [cvx.abs(self.s_cvx) <= self.s_max]
        constraints += [cvx.max(cvx.abs(self.s_cvx - self.s_prev_param)) <= self.rho]
        constraints += [cvx.max(cvx.abs(self.u_cvx - self.u_prev_param)) <= self.rho]
        prob = cvx.Problem(cvx.Minimize(objective), constraints)
        return prob

    def ode(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        s: [..., 4]  (x, θ, dx, dθ)
        u: [..., 1]  (velocity)
        returns ds/dt with shape [..., 4]
        """
        # these may be useful later
        # mp = self.pole_mass
        # mc = self.cart_mass
        L  = self.pole_length
        tau = self.tau
        g  = 9.81

        x, th, dx, dth = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
        sin_th, cos_th = torch.sin(th), torch.cos(th)

        ddx  = (u[..., 0] - dx) / tau
        ddth = -(g * sin_th + ddx * cos_th) / L

        ds = torch.stack((dx, dth, ddx, ddth), dim=-1)
        return ds
