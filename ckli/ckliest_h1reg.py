from time import perf_counter
import numpy as np
import scipy.linalg as spl
#from joblib import Parallel, delayed

def gpr(ymean, Cy, yobs, iobs):
    Cytest = Cy[iobs]
    L = spl.cholesky(Cy[np.ix_(iobs, iobs)] + np.sqrt(np.finfo(float).eps) * np.eye(iobs.size), lower=True)
    a = spl.solve_triangular(L.T, spl.solve_triangular(L, yobs - ymean[iobs], lower=True))
    V = spl.solve_triangular(L, Cytest, lower=True)
    return ymean + Cytest.T @ a, Cy - V.T @ V

def smc_gp(Ypred, CYpred, Nens, prob, rs, verbose=False, randomize_bc=False, randomize_scale=0.01):
    Nc = Ypred.size
    
    timer = perf_counter()

    Lpred = spl.cholesky(CYpred + np.sqrt(np.finfo(float).eps) * np.eye(Nc), lower=True)
    if randomize_bc:
        uens = np.vstack([prob.randomize_bc('N', randomize_scale).solve(Ypred + Lpred @ rs.randn(Nc)) for _ in range(Nens)])
    else:
        uens = np.vstack([prob.solve(Ypred + Lpred @ rs.randn(Nc)) for _ in range(Nens)])
        #with Parallel(n_jobs=8) as parallel:
        #    uens = np.vstack(parallel(delayed(prob.solve)(Ypred + Lpred @ rs.randn(Nc)) for _ in range(Nens)))

    if verbose:
        print(f'Elapsed time: {perf_counter() - timer : g} s')

    return np.mean(uens, axis=0), np.cov(uens, rowvar=False, bias=False)

def KL_via_eigh(C, Nxi):
    Nc = C.shape[0]
    Lambda, Phi = spl.eigh(C, eigvals=(Nc - Nxi, Nc - 1))
    return (Phi.real @ np.diag(np.sqrt(np.abs(Lambda))))[:, ::-1], Lambda[::-1]

class LeastSqRes(object):

    def __init__(self, NYxi, Ypred, PsiY, Nuxi, upred, Psiu, problem, Ygamma, ugamma, res_fac, L, iuobs, uobs, iYobs, Yobs, beta, ssv=None):
        self.NYxi     = NYxi
        self.Nuxi     = Nuxi
        self.problem  = problem
        self.Ypred    = Ypred
        self.PsiY     = PsiY
        self.upred    = upred
        self.Psiu     = Psiu
        self.beta12   = np.sqrt(beta)
        self.Ygamma12 = np.sqrt(Ygamma)
        self.ugamma12 = np.sqrt(ugamma)
        self.res_fac  = res_fac
        self.L        = L
        self.ssv      = ssv
        self.iuobs    = iuobs
        self.uobs     = uobs
        self.iYobs    = iYobs
        self.Yobs     = Yobs
        self.jconst   = np.block([[-self.beta12 * self.Psiu[self.iuobs], np.zeros((np.size(self.iuobs), self.NYxi))],
                                  [self.ugamma12 * (self.L @ self.Psiu), np.zeros((self.L.shape[0], self.NYxi))],
                                  [np.zeros((self.L.shape[0], self.Nuxi)), self.Ygamma12 * (self.L @ self.PsiY)]])

    def val(self, x):
        uxi = x[:self.Nuxi]
        Yxi = x[self.Nuxi:]
        u = self.upred + self.Psiu.dot(uxi)
        Y = self.Ypred + self.PsiY.dot(Yxi)
        if self.ssv is None:
            return np.concatenate((self.problem.residual(u, Y) / self.res_fac, self.beta12 * (self.uobs - u[self.iuobs]), self.ugamma12 * (self.L @ u), self.Ygamma12 * (self.L @ Y)))
        else:
            return np.concatenate((self.problem.residual(u, Y)[self.ssv] / self.res_fac, self.beta12 * (self.uobs - u[self.iuobs]), self.ugamma12 * (self.L @ u), self.Ygamma12 * (self.L @ Y)))
    
    def objectives(self, u, Y):
        return self.problem.residual(u, Y) / self.res_fac, self.beta12 * (self.uobs - u[self.iuobs]), self.ugamma12 * (self.L @ u), self.Ygamma12 * (self.L @ Y)

    def jac(self, x):
        uxi = x[:self.Nuxi]
        Yxi = x[self.Nuxi:]
        u = self.upred + self.Psiu.dot(uxi)
        Y = self.Ypred + self.PsiY.dot(Yxi)
        au = self.problem.residual_sens_u(u, Y) / self.res_fac
        aY = self.problem.residual_sens_Y(u, Y) / self.res_fac
        if self.ssv is None:
            return np.block([[au.dot(self.Psiu), aY.T.dot(self.PsiY)], [self.jconst]])
        else:
            return np.block([[au.dot(self.Psiu)[self.ssv,:], aY.T.dot(self.PsiY)[self.ssv,:]], [self.jconst]])