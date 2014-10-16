#!/usr/bin/env python
# coding: utf-8

from dolfin import *
import numpy as np
import utils
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from IPython import embed

set_log_level(ERROR)

randint = np.random.randint

# Exceptions
class RBConvergenceError(RuntimeError): pass
class LinSolvConvergenceError(RuntimeError): pass
class ParameterSpaceDepleted(RuntimeError): pass
class IllConditionedBasis(RuntimeError): pass

class AffineReducedBasisSolver(object):

    def __init__(self, forms, factors, source, function_space, param_space, bcs=None, **options):

        # Base problem is
        # Σ_i forms[i](param) * factors[i](u, v) = source(v)    ∀ v

        self.forms          = forms                    # FEniCS UFL forms
        self.factors        = factors                  # python functions
        self.source         = assemble(source, bcs=bcs).array() # FEniCS UFL form
        self.V              = function_space
        self.param_space    = param_space
        self.bcs            = bcs
        self.N              = self.source.size
        self.n              = 0

        u, v = TrialFunction(self.V), TestFunction(self.V)

        if options.has_key('inner_prod'):
            self.inner_prod_mat = sp.csr_matrix(assemble(options['inner_prod'], bcs=self.bcs).array())
        else:
            self.inner_prod_mat = sp.csr_matrix(assemble((inner(grad(u), grad(v)) + u*v)*dx, bcs=self.bcs).array())


        if len(factors) != len(forms):
            raise ValueError('`forms` and `factors` must have same length')

        self.matrices = []
        for f in self.forms:
            self.matrices.append(sp.csr_matrix(assemble(f, bcs=self.bcs).array()))

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
                          progress_plots=False,
                          symmetric=True):

        dual = method == 'ap' and not symmetric

        self.n = basis_size
        self.do_ortho = do_ortho
        self.previous_err = np.inf

        param_space_size = reduce(lambda s, l:s*len(l), self.param_space, 1)
        if self.n > param_space_size:
            raise ValueError('basis_size ({0}) must be less than the size of the parameter space ({1})'.format(self.n, param_space_size))

        self.basis = []
        if dual:
            self.basis_ = []

        self.rb_mats = [np.eye(self.n) for r in self.forms]
        self.rhs_rb = np.zeros(self.n)
        if dual:
            self.rb_mats_ = [np.eye(self.n) for r in self.forms]
            self.rhs_rb_ = np.zeros(self.n)

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

        u0 = self.spsolve(self.bl_matrix(self.s_idc[self.k]), self.source)
        u0 = self.ortho(u0)
        if dual:
            u0_ = self.spsolve(self.bl_matrix(self.s_idc[self.k]).T, -self.source)
            u0_ = self.ortho(u0_, dual=True)
        else:
            u0_ = None

        if progress_plots:
            self.plot(u0)

        if method == 'ap':
            # phi
            self.phi = self.spsolve(self.inner_prod_mat, self.source)

            # psis
            self.psis  = [] # primal
            self.psis_ = [] # dual

            self.compute_psis(u0, u0_)

        self.add(u0, u0_)

        while True:
            err, idc = self.error_argmax(method=method, dual=dual)
            if err > self.previous_err:
                msg = 'Residual norm has increased from last step (x{0}), something has gone wrong, aborting'.format(int(np.rint(err/self.previous_err)))
                if method == 'ap' and symmetric:
                    msg += '\nreduce was called with symmetric=True, if your problem is not symmetric, try switching to False or use another method for error estimation (e.g method="true")'
                raise RBConvergenceError(msg)
            self.previous_err = err
            if err < tol: break
            if self.k == self.n:
                raise RBConvergenceError('failed to meet tolerance {0:.2e} with {1} basis functions'.format(tol, self.n))

            u = self.spsolve(self.bl_matrix(idc), self.source)
            u = self.ortho(u)
            if dual:
                u_ = self.spsolve(self.bl_matrix(idc).T, -self.source)
                u_ = self.ortho(u_, dual=True)
            else:
                u_ = None
            if progress_plots:
                self.plot(u)

            if method == 'ap':
                self.compute_psis(u, u_)

            self.add(u, u_)
            self.s_idc.append(idc)

        for ub in self.basis:
            self.plot(ub)
        self.cleanup()
        return self.s_idc, self.basis

    def add(self, u, u_=None):
        self.basis.append(u)
        if u_ is not None:
            self.basis_.append(u_)
        for i, m in enumerate(self.matrices):
            for j, ub in enumerate(self.basis):
                bl_int_r = ub.dot(m*u)
                bl_int_l = u.dot(m*ub)
                self.rb_mats[i][j, self.k-1] = bl_int_r
                self.rb_mats[i][self.k-1, j] = bl_int_l
                if u_ is not None:
                    bl_int_r_ = ub.dot(m.T*u_)
                    bl_int_l_ = u_.dot(m.T*ub)
                    self.rb_mats_[i][j, self.k-1] = bl_int_r_
                    self.rb_mats_[i][self.k-1, j] = bl_int_l_
        self.rhs_rb[self.k-1] = u.dot(self.source)
        if u_ is not None:
            self.rhs_rb_[self.k-1] = -u_.dot(self.source)

    def error_argmax(self, method, dual=False):
        idc_max = None
        err_max = -1
        for idc, params in utils.ml_iterator(self.param_space, values=True):
            bump = False
            print ' '+str(idc)+'                             \r',
            if idc in self.s_idc: continue
            # solve problem on reduced basis
            M_rb = self.rb_matrix(idc)
            u_rb_coeffs = self.solve(M_rb, self.rhs_rb)
            if dual:
                M_rb_ = self.rb_matrix_(idc)
                u_rb_coeffs_ = self.solve(M_rb_, self.rhs_rb_)


            # assemble residual dual
            if method == 'ap':
                residual = np.array(self.phi)
                for n in range(self.k):
                    for q in range(len(self.forms)):
                        residual -= self.factors[q](params)*u_rb_coeffs[n]*self.psis[n][q]
                # compute error
                if dual:
                    residual_ = -np.array(self.phi)
                    for n in range(self.k):
                        for q in range(len(self.forms)):
                            residual_ = self.factors[q](params)*u_rb_coeffs_[n]*self.psis_[n][q]
                    err = np.sqrt(np.sqrt(self.inner(residual, residual)) * np.sqrt(self.inner(residual_, residual_)))
                else:
                    err = np.sqrt(self.inner(residual, residual))


            elif method == 'true':
                u_exact = self.spsolve(self.bl_matrix(idc), self.source)
                u_rb = np.zeros_like(u_exact)
                for n in range(self.k):
                    u_rb += u_rb_coeffs[n]*self.basis[n]
                err = np.sqrt(self.inner(u_exact-u_rb, u_exact-u_rb))


            if err > err_max:
                [err_max, idc_max] = [err, list(idc)]
                bump = True
            #if idc in self.s_idc:
                #print utils.color.GREEN + str(idc) + ' | '+' '.join(map(lambda x:'{0: .4f}'.format(x), u_rb_coeffs))+' || '+str(err) + utils.color.END
            #elif bump:
                #print utils.color.RED + str(idc) + ' | '+' '.join(map(lambda x:'{0: .4f}'.format(x), u_rb_coeffs))+' || '+str(err) + utils.color.END
            #else:
                #print str(idc) + ' | '+' '.join(map(lambda x:'{0: .4f}'.format(x), u_rb_coeffs))+' || '+str(err)
        if idc_max is None:
            raise ParameterSpaceDepleted
        print
        print "Max error: {0} ({1})".format(err_max, str(idc_max))
        return err_max, idc_max

    def rb_matrix(self, params_idc):
        params = [self.param_space[i][params_idc[i]] for i in range(len(params_idc))]
        M = self.factors[0](params)*self.rb_mats[0]
        for i, m in enumerate(self.rb_mats[1:], 1):
            M = M + self.factors[i](params)*m
        return M

    def rb_matrix_(self, params_idc):
        params = [self.param_space[i][params_idc[i]] for i in range(len(params_idc))]
        M = self.factors[0](params)*self.rb_mats_[0]
        for i, m in enumerate(self.rb_mats_[1:], 1):
            M = M + self.factors[i](params)*m
        return M

    def bl_matrix(self, params_idc, idx='all'):
        params = [self.param_space[i][params_idc[i]] for i in range(len(params_idc))]
        if idx == 'all':
            M = self.factors[0](params)*self.matrices[0]
            for i, m in enumerate(self.matrices[1:], 1):
                M = M + self.factors[i](params)*m
        else:
            M = self.factors[idx](params)*self.matrices[idx]
        return M

    def inner(self, u, v):
        return u.dot(self.inner_prod_mat * v)

    def ortho(self, u, dual=False):
        basis = self.basis_ if dual else self.basis
        un = np.array(u)
        if self.do_ortho:
            for ub in basis:
                un = un - self.inner(u, ub)*ub
        nrm = sqrt(self.inner(un, un))
        if nrm < 1e-10:
            raise IllConditionedBasis
        return un/nrm

    def basis_function(self):
        basis_f = map(self.fe_function, self.basis)
        return basis_f

    def fe_function(self, u):
        uf = Function(self.V)
        uf.vector()[:] = u
        return uf

    def compute_psis(self, u, u_=None):
        if self.n == self.k:
            raise RBConvergenceError
        psis_k = []
        if u_ is not None:
            psis_k_ = []
        for m in self.matrices:
            rhs = m*u
            psi_ki = self.spsolve(self.inner_prod_mat, rhs)
            psis_k.append(psi_ki)

            if u_ is not None:
                rhs = m.T*u
                psi_ki_ = self.spsolve(self.inner_prod_mat, rhs)
                psis_k_.append(psi_ki_)

        self.psis.append(psis_k)
        if u_ is not None:
            self.psis_.append(psis_k_)

    def plot(self, u):
        plot(self.fe_function(u), interactive=True)

    def cleanup(self):
        self.phi = self.psis = self.rb_mats = self.rhs_rb = None
        self.phis_ = self.rb_mats = self.rhs_rb_ = None

    @property
    def k(self):
        return len(self.basis)
