import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spl
from time import perf_counter

class DASAExp(object):

    def __init__(self, objfun, obj_sens_state, obj_sens_param, solvefun, res_sens_state, res_sens_param):
        self.objfun   = objfun
        self.solvefun = solvefun
        self.obj_sens_state = obj_sens_state
        self.obj_sens_param = obj_sens_param
        self.res_sens_state = res_sens_state
        self.res_sens_param = res_sens_param
        self.reset_timer()

    def reset_timer(self):
        self.solve_time = 0.0
        self.obj_time = 0.0
        self.dhdu_time = 0.0
        self.dhdp_time = 0.0
        self.dLdu_time = 0.0
        self.dLdp_time = 0.0
        self.adj_time = 0.0
        self.sens_time = 0.0
        
    def obj(self, p):
        time_start = perf_counter()
        self.u = self.solvefun(p)
        self.solve_time += perf_counter() - time_start
        time_start = perf_counter()
        obj = self.objfun(self.u, p)
        self.obj_time += perf_counter() - time_start
        return obj

    def grad(self, p):
        #u = self.solvefun(p)
        dhdu = self.obj_sens_state(self.u, p)
        dhdp = self.obj_sens_param(self.u, p)
        dLdu = self.res_sens_state(self.u, p)
        dLdp = self.res_sens_param(self.u, p)
        adj  = -spl.spsolve((dLdu.T).tocsc(), dhdu)
        sens = dLdp.dot(adj)
        sens = sens + dhdp
        return sens
    
class DASAExpLM(DASAExp):

    def grad(self, p):
        #u = self.solvefun(p)
        time_start = perf_counter()
        dhdu = self.obj_sens_state(self.u, p)
        self.dhdu_time += perf_counter() - time_start
        time_start = perf_counter()
        dhdp = self.obj_sens_param(self.u, p)
        self.dhdp_time += perf_counter() - time_start
        time_start = perf_counter()
        dLdu = self.res_sens_state(self.u, p)
        self.dLdu_time += perf_counter() - time_start
        time_start = perf_counter()
        dLdp = self.res_sens_param(self.u, p)
        self.dLdp_time += perf_counter() - time_start
        time_start = perf_counter()
        # dLdu = prob.A is symmetric
        adj  = -spl.spsolve(dLdu, dhdu.T)
        self.adj_time += perf_counter() - time_start
        time_start = perf_counter()
        sens = dLdp.dot(adj)
        self.sens_time += perf_counter() - time_start
        return sps.vstack([sens.T, dhdp])

class DASAExpKL(DASAExpLM):
    
    def __init__(self, objfun, obj_sens_state, obj_sens_param, solvefun, res_sens_state, res_sens_param, const_term, param_sens_coeff):
        super().__init__(objfun, obj_sens_state, obj_sens_param, solvefun, res_sens_state, res_sens_param)
        self.const_term = const_term
        self.param_sens_coeff = param_sens_coeff
    
    def obj(self, xi):
        self.p = self.const_term + self.param_sens_coeff @ xi
        return super().obj(self.p)

    def grad(self, xi):
        return super().grad(self.p) @ self.param_sens_coeff

class DASAExpLMScalar(DASAExp):

    def grad(self, p):
        #u = self.solvefun(p)
        time_start = perf_counter()
        dhdu = self.obj_sens_state(self.u, p)
        self.dhdu_time += perf_counter() - time_start
        time_start = perf_counter()
        dhdp = self.obj_sens_param(self.u, p)
        self.dhdp_time += perf_counter() - time_start
        time_start = perf_counter()
        dLdu = self.res_sens_state(self.u, p)
        self.dLdu_time += perf_counter() - time_start
        time_start = perf_counter()
        dLdp = self.res_sens_param(self.u, p)
        self.dLdp_time += perf_counter() - time_start
        time_start = perf_counter()
        # dLdu = prob.A is symmetric
        adj  = -spl.spsolve(dLdu, dhdu.T)
        self.adj_time += perf_counter() - time_start
        time_start = perf_counter()
        sens = dLdp.dot(adj)
        self.sens_time += perf_counter() - time_start
        return sens.T + dhdp

class DASAExpLMWithFlux(DASAExp):

    def __init__(self, NY, objfun, obj_sens_state, obj_sens_param, solvefun, res_sens_state, res_sens_param):
        super().__init__(objfun, obj_sens_state, obj_sens_param, solvefun, res_sens_state, res_sens_param)
        self.NY = NY

    def obj(self, p):
        time_start = perf_counter()
        self.u = self.solvefun(p[:self.NY], p[self.NY:])
        self.solve_time += perf_counter() - time_start
        time_start = perf_counter()
        obj = self.objfun(self.u, p)
        self.obj_time += perf_counter() - time_start
        return obj

    def grad(self, p):
        Y = p[:self.NY]
        #q = p[self.NY:]
        #u = self.solvefun(Y, q)
        dhdu = self.obj_sens_state(self.u, Y)
        dhdp = self.obj_sens_param(self.u, p)
        dLdu = self.res_sens_state(self.u, Y)
        dLdp = self.res_sens_param(self.u, p)
        adj  = -spl.spsolve((dLdu.T).tocsc(), dhdu.T)
        sens = dLdp.dot(adj)
        return sps.vstack([sens.T, dhdp]).toarray()