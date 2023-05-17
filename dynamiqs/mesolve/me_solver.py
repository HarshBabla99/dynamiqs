import functools

from torch import Tensor

from ..solvers.solver import Solver, depends_on_H
from ..utils.solver_utils import kraus_map


class MESolver(Solver):
    def __init__(self, *args, jump_ops: Tensor):
        super().__init__(*args)
        self.jump_ops = jump_ops[None, ...]  # (1, len(jump_ops), n, n)

    @functools.cached_property
    def sum_no_jump(self) -> Tensor:
        # -> (n, n)
        return (self.jump_ops.adjoint() @ self.jump_ops).squeeze(0).sum(dim=0)

    @depends_on_H
    def H_nh(self, t: float) -> Tensor:
        # non-hermitian Hamiltonian at time t
        # -> (b_H, 1, n, n)
        return self.H(t) - 0.5j * self.sum_no_jump

    def lindbladian(self, t: float, rho: Tensor) -> Tensor:
        H_nh_rho = self.H_nh(t) @ rho  # (b_H, b_rho, n, n)
        L_rho_Ldag = kraus_map(rho, self.jump_ops)  # (b_H, b_rho, n, n)
        return -1j * (H_nh_rho - H_nh_rho.adjoint()) + L_rho_Ldag