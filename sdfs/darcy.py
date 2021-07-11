import numpy as np
import scipy.sparse.linalg as spl
import scipy.sparse as sps
from sdfs.tpfa import TPFA
import time

class DarcyExp(object):

    def __init__(self, tpfa, ssv=None):
        self.tpfa = tpfa
        self.ssv = range(self.tpfa.geom.cells.num) if ssv is None else ssv
        self.Nc = self.tpfa.geom.cells.num
        self.Nc_range = np.arange(self.Nc)
        self.rows = np.repeat(self.Nc_range, 5)
        neumann_bc = (self.tpfa.bc.kind == 'N')
        Nq = np.count_nonzero(neumann_bc)
        self.dLdq = sps.csc_matrix((-np.ones(Nq), (np.arange(Nq), self.tpfa.geom.cells.to_hf[2*self.tpfa.Ni:][neumann_bc])), shape=(Nq, self.Nc))

    def randomize_bc(self, kind, scale):
        self.tpfa.bc.randomize(kind, scale)
        self.tpfa.update_rhs(kind)
        return self

    def increment_bc(self, kind, value):
        self.tpfa.bc.increment(kind, value)
        self.tpfa.update_rhs(kind)
        return self

    def solve(self, Y, q=None):
        self.A, b = self.tpfa.ops(np.exp(Y), q)
        return spl.spsolve(self.A, b)

    def residual(self, u, Y):
        self.A, b = self.tpfa.ops(np.exp(Y))
        return self.A @ u - b

    def residual_sens_Y(self, u, Y):
        # call residual(self, u, Y) before residual_sens_Y(self, u, Y)
        Nc = self.tpfa.geom.cells.num
        diag, offdiags, cols, bsens = self.tpfa.sens_old()
        vals = np.hstack(((diag * u - (offdiags * u[cols]).sum(axis=1) - bsens)[:, None],
            (u[cols] - u[:, None]) * offdiags)) * np.exp(Y)[:, None]
        cols = np.hstack((self.Nc_range[:, None], cols))
        keep = np.flatnonzero(cols >= 0)
        return sps.csc_matrix((vals.ravel()[keep], (self.rows[keep], cols.ravel()[keep])), shape=(Nc, Nc))
        
    def residual_sens_u(self, u, Y):
        # call residual(self, u, Y) before residual_sens_u(self, u, Y)
        return self.A

    def residual_sens_p(self, u, p):
        # call residual(self, u, Y) before residual_sens_p(self, u, p)
        return sps.vstack([self.residual_sens_Y(u, p[:self.tpfa.geom.cells.num]), self.dLdq])


class DarcyExpTimeDependent(DarcyExp):

    def __init__(self, tpfa, ss, dt, ssv=None):
        super().__init__(tpfa, ssv)
        self.ss = ss
        self.dt = dt
        self.c = self.ss / self.dt
        self.C = self.c * sps.eye(self.Nc)
        self.prev_u = np.zeros(self.Nc)
        self.c_prev_u = self.c * self.prev_u
    
    def update_u(self, prev_u):
        self.prev_u = prev_u
        self.c_prev_u = self.c * self.prev_u

    def solve(self, Y, q=None):
        self.A, b = self.tpfa.ops(np.exp(Y), q)
        return spl.spsolve(self.A - self.C, b - self.c_prev_u)

    def residual(self, u, Y):
        self.A, b = self.tpfa.ops(np.exp(Y))
        return self.A @ u - b - self.c * (u - self.prev_u)

    def residual_sens_u(self, u, Y):
        return self.A - self.C