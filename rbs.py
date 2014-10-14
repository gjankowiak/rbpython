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
        self.L2_prod = sp.csr_matrix(assemble(u*v*dx, bcs=self.bcs).array())

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
    def reduce(self, tol, method='ap', basis_size=10, do_ortho=True, start_idc=None):

        self.n = basis_size
        self.do_ortho = do_ortho

        param_space_size = reduce(lambda s, l:s*len(l), self.param_space, 1)
        if self.n > param_space_size:
            raise ValueError('basis_size ({0}) must be less than the size of the parameter space ({1})'.format(self.n, param_space_size))

        self.basis = []

        self.rb_mats = [np.eye(self.n) for r in self.forms]
        self.rhs_rb = np.zeros(self.n)

        if method != 'ap':
            raise NotImplemented('only a posteriori estimator is available for now')

        self.s_idc = []

        # initialization
        if start_idc is None:
            i_idc = []
            for s in self.param_space:
                i_idc.append(randint(len(s)))
        else:
            i_idc = start_idc
        self.s_idc.append(i_idc)

        u0 = self.spsolve(self.bl_matrix(self.s_idc[self.k]), self.source)
        u0 = self.ortho(u0)

        # phi
        self.phi = self.spsolve(self.inner_prod_mat, self.source)

        # psis
        self.psis = []
        self.compute_psis(u0)

        self.add(u0)

        while True:
            err, idc = self.error_argmax()
            if err < tol: break
            if self.k == self.n:
                raise RBConvergenceError('failed to meet tolerance {0:.2e} with {1} basis functions'.format(tol, self.n))

            u = self.spsolve(self.bl_matrix(idc), self.source)
            u = self.ortho(u)

            self.compute_psis(u)
            self.add(u)
            self.s_idc.append(idc)

        for ub in self.basis:
            self.plot(ub)
        self.cleanup()
        return self.s_idc, self.basis

    def add(self, u):
        self.basis.append(u)
        for i, m in enumerate(self.matrices):
            for j, ub in enumerate(self.basis):
                bl_int = ub.dot(m*u)
                # not very efficient
                self.rb_mats[i][j, self.k-1] = bl_int
                self.rb_mats[i][self.k-1, j] = bl_int
        self.rhs_rb[self.k-1] = u.dot(self.source)

    def error_argmax(self):
        idc_max = None
        err_max = -1
        for idc, params in utils.ml_iterator(self.param_space, values=True):
            if idc in self.s_idc: continue
            # solve problem on reduced basis
            M_rb = self.rb_matrix(idc)
            u_rb = self.solve(M_rb, self.rhs_rb)
            # assemble residual dual
            residual = np.array(self.phi)
            for n in range(self.k):
                for q in range(len(self.forms)):
                    residual -= self.factors[q](params)*u_rb[n]*self.psis[n][q]
            # compute error
            err = np.sqrt(self.inner(residual, residual))
            if err > err_max:
                [err_max, idc_max] = [err, list(idc)]
        if idc_max is None:
            raise ParameterSpaceDepleted
        print "Max error: {0}".format(err_max)
        return err_max, idc_max

    def rb_matrix(self, params_idc):
        params = [self.param_space[i][params_idc[i]] for i in range(len(params_idc))]
        M = self.factors[0](params)*self.rb_mats[0]
        for i, m in enumerate(self.rb_mats[1:], 1):
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

    def ortho(self, u):
        un = np.array(u)
        if self.do_ortho:
            for ub in self.basis:
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

    def compute_psis(self, u):
        if self.n == self.k:
            raise RBConvergenceError
        psis_k = []
        for m in self.matrices:
            rhs = m*u
            psi_ki = self.spsolve(self.inner_prod_mat, rhs)
            psis_k.append(psi_ki)
        self.psis.append(psis_k)

    def plot(self, u):
        f_u = Function(self.V)
        f_u.vector()[:] = u
        plot(f_u, interactive=True)

    def cleanup(self):
        self.phi = self.psis = self.rb_mats = self.rhs_rb = None

    @property
    def k(self):
        return len(self.basis)
