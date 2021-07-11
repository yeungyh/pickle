import numpy as np
import scipy.linalg as spl

def gpr(ymean, Cy, yobs, iobs):
    tol = np.sqrt(np.finfo(float).eps)
    Cyobs = Cy[np.ix_(iobs, iobs)]
    di = np.diag_indices_from(Cyobs)
    Cyobs[di] = Cyobs[di] + tol
    Cytest = Cy[iobs]
    L = spl.cholesky(Cyobs, lower=True)
    a = spl.solve_triangular(L.T, spl.solve_triangular(L, yobs - ymean[iobs], lower=True))
    V = spl.solve_triangular(L, Cytest, lower=True)
    ypred = ymean + Cytest.T.dot(a)
    Cypred = Cy - V.T.dot(V)
    return ypred, Cypred

def smc_gp(Ypred, CYpred, Nens, solver, verbose=False):
    Nc = Ypred.size
    uens = np.zeros((Nens, Nc))

    timer = time()

    tol = np.sqrt(np.finfo(float).eps)
    CYpred_tol = CYpred
    di  = np.diag_indices_from(CYpred_tol)
    CYpred_tol[di] = CYpred_tol[di] + tol

    Lpred = spl.cholesky(CYpred_tol, lower=True)

    for ie in range(Nens):
        Ys = Ypred + Lpred.dot(npr.randn(Nc))
        us = solver(Ys)
        uens[ie] = us

    if verbose == True:
        print("Elapsed time: {:g} s".format(time() - timer))

    umean = np.mean(uens, axis=0)
    Cu = np.cov(uens, rowvar=False, bias=False)

    return umean, Cu

    
