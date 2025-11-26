"""
Cart velocity (servo) controlled cartpole.
"""

from scp_solver import SCPSolver, SolverParams

import cvxpy as cvx
import torch
import numpy as np
from cartpole import CartpoleEnvironmentParams
from cartpole_energy import CartpoleEnergy

class CartpoleSolverVelocity(SCPSolver):
    ep: CartpoleEnvironmentParams
    energy: CartpoleEnergy
    def __init__(self, sp: SolverParams, ep: CartpoleEnvironmentParams):
        super().__init__(sp)
        self.ep = ep
        self.g = 9.81
        self.energy = CartpoleEnergy(ep)
        self.setup()

    def opt_problem(self) -> cvx.Problem:
        # set up cvxpy optimization
        objective = cvx.quad_form((self.s_cvx[self.N] - self.s_goal), self.params.P)
        objective += cvx.sum([cvx.sum(cvx.huber(self.params.Q @ (self.s_cvx[i] - self.s_goal))) + cvx.quad_form(self.u_cvx[i], self.params.R) for i in range(self.N)])
        #objective = cvx.quad_form((self.s_cvx[self.N] - self.s_goal), self.P) + cvx.sum(
        #    [cvx.quad_form(self.s_cvx[i] - self.s_goal, self.Q) + cvx.quad_form(self.u_cvx[i], self.R) for i in range(self.N)])
        constraints = [self.s_cvx[i + 1] == self.c_param[i] + self.A_param[i] @ self.s_cvx[i] +
                       self.B_param[i] @ self.u_cvx[i] for i in range(self.N)]
        constraints += [self.s_cvx[0] == self.s0]
        constraints += [cvx.abs(self.u_cvx) <= self.params.u_max]
        constraints += [cvx.abs(self.s_cvx) <= self.params.s_max]
        constraints += [cvx.max(cvx.abs(self.s_cvx - self.s_prev_param)) <= self.params.rho]
        constraints += [cvx.max(cvx.abs(self.u_cvx - self.u_prev_param)) <= self.params.rho]
        prob = cvx.Problem(cvx.Minimize(objective), constraints)
        return prob

    def ode(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        s: [..., 4]  (x, θ, dx, dθ)
        u: [..., 1]  (velocity)
        returns ds/dt with shape [..., 4]
        """
        # these may be useful later
        # m_p = self.pole_mass
        # m_c = self.cart_mass
        L  = self.ep.pole_length
        tau = self.ep.cart_tau
        g  = self.g

        x, th, dx, dth = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
        sin_th = torch.sin(th)
        cos_th = torch.cos(th)

        ddx  = (u[..., 0] - dx) / tau
        # mass at tip of pole
        #ddth = -(g * sin_th + ddx * cos_th) / L
        # mass along pole
        ddth = -1.5 * (g * sin_th + ddx * cos_th) / L

        ds = torch.stack((dx, dth, ddx, ddth), dim=-1)
        return ds
