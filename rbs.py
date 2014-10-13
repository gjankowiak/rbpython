#!/usr/bin/env python
# coding: utf-8

from dolfin import *
import numpy as np
import utils

from IPython import embed

parameters['linear_algebra_backend'] = 'uBLAS'
set_log_level(ERROR)

randint = np.random.randint

class AffineReducedBasisSolver(object):

    def __init__(self, forms, factors, source, function_space, param_space, **options):

        # Base problem is
        # Σ_i forms[i](param) * factors[i](u, v) = source(v)    ∀ v

        self.forms          = forms            # FEniCS UFL forms
        self.factors        = factors          # python functions
        self.source         = assemble(source) # FEniCS UFL form
        self.V              = function_space
        self.param_space    = param_space
        self.N              = self.source.size()
        self.n              = 10

        u, v = TrialFunction(self.V), TestFunction(self.V)
        self.L2_prod        = assemble(u*v*dx)

        if options.has_key('inner_prod'):
            self.inner_prod = assemble(options['inner_prod'])
        else:
            self.inner_prod = assemble((inner(grad(u), grad(v)) + u*v)*dx)


        if len(factors) != len(forms):
            raise ValueError('`forms` and `factors` must have same length')

        self.matrices = []
        for f in self.forms:
            self.matrices.append(assemble(f))


    # methods:
    #   - 'ap'   : a posteriori estimator
    #   - 'pod'  : proper orthogonal decomposition
    #   - 'true' : true error
    def reduce(self, tol, method='ap', n=10):

        self.n = n

        self.s = uBLASKrylovSolver() #LinearSolver()
        self.s.parameters["monitor_convergence"] = True
        self.s.parameters['absolute_tolerance'] = 1e-13
        self.basis = []

        self.rb_mats = [self.matrix(self.n) for r in self.forms]
        for rb_mat in self.rb_mats:
            rb_mat.ident(np.array(range(self.n), dtype=np.intc))
        self.rb_rhs = self.vector(self.n)

        if method != 'ap':
            raise NotImplemented('only a posteriori estimator is available for now')

        self.s_idc = [[] for r in self.param_space]

        # initialization
        i_idx = []
        for s in self.param_space:
            i_idx.append(randint(len(s)))
        self.s_idc.append(i_idx)

        u0 = self.vector()
        self.s.solve(self.bl_matrix(self.s_idc[self.k]), u0, self.source)
        u0 = self.ortho(u0)

        # phi
        self.phi = self.vector()
        self.s.solve(self.inner_prod, self.phi, self.source)

        # psis
        self.psis = []
        self.compute_psis(u0)

        self.add(u0)

        while True:
            err, idc = self.error_argmax()
            if err < tol: break

            u = self.vector()
            self.s.solve(self.bl_matrix(idc), u, self.source)
            u = self.ortho(u)

            self.compute_psis(u)
            self.add(u)

    def add(self, u):
        self.basis.append(u)
        _ = self.vector(u.size())
        for i, m in enumerate(self.matrices):
            for j, ub in enumerate(self.basis):
                m.mult(u, _)
                bl_int = ub.inner(_)
                # not very efficient
                self.rb_mats[i].set(utils.f64_array(bl_int), utils.intc_array([j]), utils.intc_array([self.k]))
        self.L2_prod.mult(u, _)
        self.rb_rhs.add(utils.f64_array(u.inner(_)), utils.intc_array([self.k-1]))

    def error_argmax(self):
        u_rb_comp = self.vector(self.n)
        idc_max = None
        err_max = -1
        for idc, params in utils.ml_iterator(self.param_space, values=True):
            if idc in self.s_idc: continue
            # solve problem on reduced basis
            M_rb = self.rb_matrix(idc)
            self.s.solve(M_rb, u_rb_comp, self.rb_rhs)
            u_rb_arr = u_rb_comp.array()
            # assemble residual dual
            residual = uBLASVector(self.phi)
            for n in range(self.k):
                for q in range(len(self.forms)):
                    residual.add_local(self.factors[q](params)*u_rb_arr[n]* self.psis[n][q])
            # compute error
            err = np.sqrt(self.inner(residual, residual))
            if err > err_max:
                [err_max, idc_max] = [err, idc]
        print "Current error: {0}".format(err_max)
        return err_max, idc_max

    def _test_solve(self, mat):
        n = mat.size(0)
        xx = self.vector(n)
        rhs = self.vector(n)
        rhs.array()[:] = np.random.random(n)
        self.s.solve(mat, xx, rhs)

    def rb_matrix(self, params_idc):
        params = [self.param_space[i][params_idc[i]] for i in range(len(params_idc))]
        M = self.factors[0](params)*self.rb_mats[0]
        self._test_solve(self.rb_mats[0])
        for i, m in enumerate(self.rb_mats[1:]):
            M += self.factors[i](params)*m
            self._test_solve(M)
        return M

    def bl_matrix(self, params_idc, idx='all'):
        params = [self.param_space[i][params_idc[i]] for i in range(len(params_idc))]
        if idx == 'all':
            M = self.factors[0](params)*self.matrices[0]
            for i, m in enumerate(self.matrices[1:]):
                M += self.factors[i](params)*m
        else:
            M = self.factors[idx](params)*self.matrices[idx]
        return M

    def vector(self, size=0):
        return uBLASVector(size or self.N)

    def matrix(self, size=0):
        return uBLASDenseMatrix(size or self.N, size or self.N)

    def inner(self, u, v):
        _ = self.vector(u.size())
        self.inner_prod.mult(v, _)
        return u.inner(_)

    def ortho(self, u):
        un = self.vector(u)
        for ub in self.basis:
            un = un - self.inner(u, ub)*ub
        un = un/sqrt(self.inner(un, un))
        return un

    def basis_function(self):
        basis_f = map(self.fe_function, self.basis)
        return basis_f

    def fe_function(self, u):
        uf = Function(self.V)
        uf.vector()[:] = u
        return uf

    def compute_psis(self, u):
        if self.n == self.k:
            raise IndexError('Bilinear integral matrices are full, try calling reduce with a higher value for n')
        psis_k = []
        for m in self.matrices:
            psi_ki = self.vector()
            rhs = self.vector()
            m.mult(-u, rhs)
            self.s.solve(self.inner_prod, psi_ki, rhs)
            psis_k.append(psi_ki)
        self.psis.append(psis_k)

    @property
    def k(self):
        return len(self.basis)
