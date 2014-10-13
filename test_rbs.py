from dolfin import *
import rbs
import numpy as np

parameters['linear_algebra_backend'] = 'uBLAS'

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "P", 1)

u, v = TrialFunction(V), TestFunction(V)

a0 = inner(grad(u), grad(v))*dx
a1 = u*v*dx
source = v*dx

def f0(params):
    return 1

def f1(params):
    return params[0]

param_space = [np.linspace(0, 1, 1000)]

solvr = rbs.AffineReducedBasisSolver([a0, a1], [f0, f1], source, V, param_space)
solvr.reduce(1e-8)
