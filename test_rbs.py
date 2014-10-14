from dolfin import *
import rbs
import numpy as np

parameters['linear_algebra_backend'] = 'uBLAS'

mesh = UnitSquareMesh(50, 50)
V = FunctionSpace(mesh, "P", 1)

u, v = TrialFunction(V), TestFunction(V)

a0 = inner(grad(u), grad(v))*dx
a1 = u*v*dx
f  = Expression('10*sin(3*x[0])')
source = f*v*dx

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0.0), boundary)

def f0(params):
    return 1

def f1(params):
    return params[0]

param_space = [np.linspace(1, 1000, 500)]

solvr = rbs.AffineReducedBasisSolver([a0, a1], [f0, f1], source, V, param_space, bcs=bc)
solvr.reduce(1e-8, do_ortho=True, basis_size=20, start_idc=[0])
