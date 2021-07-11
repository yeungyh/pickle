import numpy as np
import scipy.linalg as spl
import scipy.sparse as sps

def compute_Lreg(geom):
    Nc    = geom.cells.num
    Nfint = geom.faces.num_interior
    return sps.csc_matrix((np.concatenate((-np.ones(Nfint), np.ones(Nfint))), (np.tile(np.arange(Nfint), 2), geom.faces.neighbors[:, :Nfint].ravel())), shape=(Nfint, Nc))

class LossVec(object):

    def __init__(self, Nu, NY, iuobs, uobs, iYobs, Yobs, gamma, L):
        self.Nu = Nu
        self.NY = NY
        self.iuobs = iuobs
        self.uobs  = uobs
        self.iYobs = iYobs
        self.Yobs  = Yobs
        self.gamma12 = np.sqrt(gamma)
        self.L     = L
        self.grad_u_matrix = sps.csr_matrix((-np.ones(self.iuobs.size), (np.arange(self.iuobs.size), self.iuobs)), shape=(self.iuobs.size, self.Nu))
        self.grad_Y_matrix = sps.vstack([sps.coo_matrix((-np.ones(self.iYobs.size), (np.arange(self.iYobs.size), self.iYobs)), shape=(self.iYobs.size, self.NY)),
                           self.gamma12 * self.L])

    def val(self, u, Y):
        return np.concatenate((self.uobs - u[self.iuobs], self.Yobs - Y[self.iYobs], self.gamma12 * (self.L @ Y)))

    def grad_u(self, u, Y):
        return self.grad_u_matrix

    def grad_Y(self, u, Y):
        return self.grad_Y_matrix

class LossVecWithFlux(LossVec):

    def __init__(self, Nu, NY, Nq, iuobs, uobs, iYobs, Yobs, gamma, L):
        super().__init__(Nu, NY, iuobs, uobs, iYobs, Yobs, gamma, L)
        self.Nq = Nq
        self.grad_p_matrix = sps.block_diag((self.grad_Y_matrix, sps.dia_matrix((self.gamma12 * np.ones(Nq), [0]), shape=(self.Nq, self.Nq))))

    def val(self, u, p):
        return np.concatenate((self.uobs - u[self.iuobs], self.Yobs - p[self.iYobs], self.gamma12 * (self.L @ p[:self.NY]), self.gamma12 * p[self.NY:]))

    def grad_p(self, u, p):
        return self.grad_p_matrix

class LossScalar(object):

    def __init__(self, Nu, NY, iuobs, uobs, iYobs, Yobs, gamma, L):
        self.Nu = Nu
        self.NY = NY
        self.iuobs = iuobs
        self.uobs  = uobs
        self.iYobs = iYobs
        self.Yobs  = Yobs
        self.gamma12 = np.sqrt(gamma)
        self.L     = L
        self.grad_u_matrix = sps.csc_matrix((-np.ones(self.iuobs.size), (np.arange(self.iuobs.size), self.iuobs)), shape=(self.iuobs.size, self.Nu))
        self.grad_Y_matrix = sps.csc_matrix((-np.ones(self.iYobs.size), (np.arange(self.iYobs.size), self.iYobs)), shape=(self.iYobs.size, self.NY))

    def val(self, u, Y):
        return spl.norm(self.uobs - u[self.iuobs], 2) + spl.norm(self.Yobs - Y[self.iYobs], 2) + self.gamma12 * spl.norm(self.L @ Y, 2)

    def grad_u(self, u, Y):
        return -self.grad_u_matrix.T @ (self.uobs - u[self.iuobs])

    def grad_Y(self, u, Y):
        return -self.grad_Y_matrix.T @ (self.Yobs - Y[self.iYobs]) + self.gamma12 * self.L.T @ (self.L @ Y)