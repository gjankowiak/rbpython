from dolfin import *
import rbs
import numpy as np

parameters['linear_algebra_backend'] = 'uBLAS'

Nf = 100
penalization = Constant(1e8)
damping = Constant(1e8)

MCell = UnitSquareMesh(Nf, Nf)
MCell.coordinates()[:] = MCell.coordinates()

def boundary(x, on_boundary):
    return on_boundary


# P1 elements for the density
VCell  = FunctionSpace(MCell, "P", 1)

# Vector P1 elements for the convection term
VCellV = VectorFunctionSpace(MCell, "P", 1)

bc = DirichletBC(VCell, Constant(0.0), boundary)

buildings = Expression('(pow((c*(x[0]+shift_x) - floor(c*(x[0]+shift_x))-0.5),2) + pow(c*(x[1]+shift_y) - floor(c*(x[1]+shift_y))-0.5, 2)) > r', r=0.04, c=10, shift_x=0, shift_y=0)
streets = 1-buildings
wind = Expression(['ws', 'c*ws'], ws=1, c=5, element=VCellV.ufl_element())

wind.ws = 2
wind.c  = 1
buildings.c=2

f = Expression('exp(-l*(x[0]+shift_x))', l=1, shift_x=0, shift_y=0)
f.l=1

u = TrialFunction(VCell)
v = TestFunction(VCell)

a0 = (streets+buildings*penalization)*inner(grad(u), grad(v))*dx
a1 = damping*buildings*u*v*dx
a2 = streets*u*v*dx
a3 = streets*u.dx(0)*v*dx
a4 = streets*u.dx(1)*v*dx

f0 = lambda p:1
f1 = lambda p:1
f2 = lambda p:p[0]
f3 = lambda p:p[1]
f4 = lambda p:p[2]

n = 4
forms = [a0, a1, a2, a3, a4][:n]
factors = [f0, f1, f2, f3, f4][:n]

source = f*v*dx

param_space = [
        np.linspace(10, 100, 100),
        np.linspace(-50, 50, 100),
#        np.linspace(-50, 50, 100)
        ]

solvr = rbs.AffineReducedBasisSolver(forms, factors, source, VCell, param_space, bcs=bc)
solvr.reduce(1e-10, do_ortho=True, basis_size=30, progress_plots=False, method='ap', symmetric=False)
