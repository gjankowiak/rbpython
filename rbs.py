#!/usr/bin/env python
# coding: utf-8

from dolfin import *
import numpy as np
import utils
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from time import time

from IPython import embed

set_log_level(ERROR)

randint = np.random.randint

# Exceptions
class RBConvergenceError(RuntimeError): pass
class LinSolvConvergenceError(RuntimeError): pass
class ParameterSpaceDepleted(RuntimeError): pass
class IllConditionedBasis(RuntimeError): pass

class AffineReducedBasisSolver(object):

    def __init__(self, forms, coeffs, factors, source, function_space, param_space, bcs=None, **options):

        # Base problem is
        # Σ_i forms[i](param) * factors[i](u, v) = source(v)    ∀ v

        self.forms          = forms                    # FEniCS UFL forms
        self.coeffs         = coeffs
        self.factors        = factors                  # python functions
        self.source         = source                   # FEniCS UFL form
        self.V              = function_space
        self.param_space    = param_space
        self.bcs            = bcs
        self.n              = 0

        u, v = TrialFunction(self.V), TestFunction(self.V)

        if options.has_key('inner_prod'):
            self.inner_prod = options['inner_prod']
        else:
            self.inner_prod = (inner(grad(u), grad(v)) + u*v)*dx
        self.inner_prod_mat = assemble(self.inner_prod, bcs=self.bcs)

        if len(coeffs) != len(factors):
            raise ValueError('`forms` and `factors` must have same length')

    def spsolve(self, A, rhs):
        sol = spla.spsolve(A, rhs)
        return sol

    def solve(self, A, rhs):
        sol = la.solve(A, rhs)
        return sol

    # methods:
    #   - 'ap'   : a posteriori estimator
    #   - 'pod'  : proper orthogonal decomposition
    #   - 'true' : true error
    def reduce(self, tol, method='ap',
                          basis_size=10,
                          do_ortho=True,
                          start_idc=None,
                          progress_plots=False):

        self.n = basis_size
        self.do_ortho = do_ortho
        self.previous_err = np.inf

        self.errors = []

        param_space_size = reduce(lambda s, l:s*len(l), self.param_space, 1)
        if self.n > param_space_size:
            raise ValueError('basis_size ({0}) must be less than the size of the parameter space ({1})'.format(self.n, param_space_size))

        self.basis = []

        self.rb_mats = [np.eye(self.n) for r in self.coeffs]
        self.rhs_rb = np.zeros(self.n)

        self.s_idc = []

        # initialization
        if start_idc is None:
            i_idc = []
            for s in self.param_space:
                i_idc.append(randint(len(s)))
        else:
            i_idc = start_idc
        print 'Starting with indices: '+str(i_idc)
        self.s_idc.append(i_idc)

        u0 = Function(self.V)
        solve(self.bl_matrix(i_idc) == self.source, u0, bcs=self.bcs)
        u0 = self.ortho(u0)

        if progress_plots:
            self.plot(u0)

        if method in ['ap', 'sap']:
            # phi
            phi = Function(self.V)
            solve(self.inner_prod == self.source, phi, bcs=self.bcs)
            self.phi = phi

            # psis
            self.psis  = [] # primal

            self.compute_psis(i_idc, u0)

        self.add(i_idc, u0)

        while True:
            err, idc = self.error_argmax(method=method)
            #if err > self.previous_err:
            #    msg = 'Residual norm has increased from last step (x{0}), something has gone wrong, aborting'.format(int(np.rint(err/self.previous_err)))
            #    return self.errors, RBConvergenceError(msg)
            self.previous_err = err
            if err < tol: break
            if self.k == self.n:
                return self.errors, RBConvergenceError('failed to meet tolerance {0:.2e} with {1} basis functions'.format(tol, self.n))

            u = Function(self.V)
            solve(self.bl_matrix(idc) == self.source, u, bcs=self.bcs)
            if progress_plots:
                self.plot(u)
            u = self.ortho(u)

            if method in ['ap', 'sap']:
                self.compute_psis(idc, u)

            self.add(idc, u)
            self.s_idc.append(idc)

        for ub in self.basis:
            self.plot(ub)
        self.cleanup()
        return self.errors, None

    def add(self, params_idc, u):
        self.basis.append(u)
        for i, m in enumerate(self.rb_mats):
            bl_m = assemble(self.bl_matrix(params_idc, idx=i, bare=True))
            for j, ub in enumerate(self.basis):
                bl_int_r = ub.vector().inner(bl_m*u.vector())
                bl_int_l = u.vector().inner(bl_m*ub.vector())
                m[j, self.k-1] = bl_int_r
                m[self.k-1, j] = bl_int_l

        self.rhs_rb[self.k-1] = u.vector().inner(assemble(self.source))

    def error_argmax(self, method):
        idc_max = None
        params_max = None
        err_max = -1
        ap_err_max = -1
        residual = Function(self.V)
        for idc, params in utils.ml_iterator(self.param_space, values=True):
            residual.vector().zero()
            bump = False
            print '{0} ({1})                   \r'.format(idc_max, idc),
            if idc in self.s_idc: continue
            # solve problem on reduced basis
            M_rb = self.rb_matrix(idc)[:self.k, :self.k]
            u_rb_coeffs = self.solve(M_rb, self.rhs_rb[:self.k])

            # assemble residual
            if method in ['ap', 'sap']:

                v = Vector(self.phi.vector())
                for q in range(len(self.coeffs)):
                    factor = self.factors[q](params)
                    for n in range(self.k):
                        v.axpy(-factor*u_rb_coeffs[n], self.psis[n][q].vector())
                # compute error
                residual.vector()[:] = v
                ap_err = err = np.sqrt(self.inner(residual, residual))

            if method in ['true', 'sap']:
                u_exact = Function(self.V)
                solve(self.bl_matrix(idc) == self.source, u_exact, bcs=self.bcs)
                for n in range(self.k):
                    residual.vector()[:] += u_rb_coeffs[n]*self.basis[n].vector()
                residual.vector()[:] = u_exact.vector() - residual.vector()
                true_err = err = np.sqrt(self.inner(residual, residual))

            if err > err_max and not idc in self.s_idc:
                [err_max, idc_max, params_max] = [err, list(idc), params]
                if method == 'sap':
                    ap_err_max = ap_err
                bump = True

            #msg = '{0} {1} || {2}'.format(str(idc), ' '.join(map(lambda x:'{0: .4f}'.format(x), u_rb_coeffs)), err)
            #msg = '{0} || {2} ({3})'.format(str(idc), ' '.join(map(lambda x:'{0: .4f}'.format(x), u_rb_coeffs)), err, np.median(toc-tic))
            #msg = '{0} || {2}'.format(str(idc), ' '.join(map(lambda x:'{0: .4f}'.format(x), u_rb_coeffs)), err)
            #if idc in self.s_idc:
            #    print utils.color.GREEN + msg + utils.color.END
            #elif bump:
            #    print utils.color.RED + msg + utils.color.END
            #else:
            #    print msg
        if idc_max is None:
            raise ParameterSpaceDepleted
        print
        print "Max error: {0} ({1}, {2})".format(err_max, str(idc_max), str(params_max))
        if method == 'sap':
            self.errors.append(ap_err_max)
        else:
            self.errors.append(err_max)
        return err_max, idc_max

    def rb_matrix(self, params_idc):
        params = [self.param_space[i][params_idc[i]] for i in range(len(params_idc))]
        M = self.factors[0](params)*self.rb_mats[0]
        for i, m in enumerate(self.rb_mats[1:], 1):
            M = M + self.factors[i](params)*m
        return M

    def bl_matrix(self, params_idc, idx='all', bare=False):
        params = [self.param_space[i][params_idc[i]] for i in range(len(params_idc))]
        for i, c in enumerate(self.coeffs):
            if idx == i or idx == 'all':
                val = 1 if bare else self.factors[i](params)
            else:
                val = 0.0
            c.assign(val)
        return self.forms

    def inner(self, u, v):
        return u.vector().inner(self.inner_prod_mat * v.vector())

    def ortho(self, u):
        basis = self.basis
        un = Function(u)
        if self.do_ortho:
            for ub in basis:
                un.vector()[:] = un.vector() - self.inner(u,ub)*ub.vector()
        nrm = sqrt(self.inner(un, un))
        if nrm < 1e-10:
            raise IllConditionedBasis
        un.vector()[:] = un.vector() / nrm
        return un

    def basis_function(self):
        basis_f = map(self.fe_function, self.basis)
        return basis_f

    def fe_function(self, u):
        uf = Function(self.V)
        uf.vector()[:] = u
        return uf

    def compute_psis(self, params_idc, u):
        if self.n == self.k:
            raise RBConvergenceError
        psis_k = []
        for i, f in enumerate(self.coeffs):
            rhs = assemble(self.bl_matrix(params_idc, idx=i, bare=True)) * u.vector()
            A = self.inner_prod_mat
            self.bcs.apply(A, rhs)
            psi_ki = Function(self.V)
            solve(A, psi_ki.vector(), rhs)
            psis_k.append(psi_ki)

        self.psis.append(psis_k)

    def plot(self, u):
        plot(u, mode="color", interactive=True)

    def cleanup(self):
        self.phi = self.psis = self.rb_mats = self.rhs_rb = None

    @property
    def k(self):
        return len(self.basis)
