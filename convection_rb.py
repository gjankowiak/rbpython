from dolfin import *
import rbs
import numpy as np
import matplotlib.pyplot as plt

parameters['linear_algebra_backend'] = 'uBLAS'

Nf = 80
penalization = Constant(1e8)
damping = Constant(1e8)

domain = Rectangle(0, 0, 1, 1) - Circle(0.25, 0.25, 0.1) - Circle(0.25, 0.75, 0.1) - Circle(0.75, 0.25, 0.1) - Circle(0.75, 0.75, 0.1)
MCell = Mesh(domain, Nf)

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

a0 = inner(grad(u), grad(v))*dx
a1 = u*v*dx
a2 = u.dx(0)*v*dx
a3 = u.dx(1)*v*dx
forms = [a0, a1, a2, a3]

[c0, c1, c2, c3] = [Constant(0.0) for i in range(4)]
coeffs = [c0, c1, c2, c3]

f0 = lambda p:1
f1 = lambda p:1
f2 = lambda p:p[0]*cos(p[1])
f3 = lambda p:p[0]*sin(p[1])
factors = [f0, f1, f2, f3]

full_form = sum(map(lambda i:i[0]*i[1], zip(coeffs, forms)))
full_form = c0*a0 + c1*a1 + c2*a2 + c3*a3

source = f*v*dx

param_space = [
        np.linspace(0, 50, 20),
        np.linspace(0, 2*pi, 20),
        ]

solvr = rbs.AffineReducedBasisSolver(full_form, coeffs, factors, source, VCell, param_space, bcs=bc)
errs, exception = solvr.reduce(1e-10, do_ortho=True, basis_size=50, progress_plots=False, method='ap')

plt.semilogy(errs)
plt.show()

if exception:
    raise exception
