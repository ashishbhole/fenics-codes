"""
1) FEniCS tutorial demo program: Linear elastic problem.

  -div(sigma(u)) = f

The model is used to simulate an elastic beam clamped at
its left end and deformed under its own weight.

2) Thermodynamically consistent derivation and computation of electro-thermo-mechanical systems for solid bodies
by B. E. Abali and F. A. Reich: Transformation from Voigt to Tensor notations.
"""

from __future__ import print_function
from fenics import *
import ufl as ufl
import numpy as np
# modules created locally
import voigt_to_tensor as vt
import mechanical_properties as mp

# No of points in each direction
N = 5

# domain size
Xmin = Ymin = Zmin = 0.0
Xmax = Ymax = Zmax = 1.0

# Variable for indices offered by dolfin for tensor algebra
i, j, k, l = ufl.indices(4)

# Create mesh and define function space
mesh = BoxMesh(Point(Xmin, Ymin, Zmin), Point(Xmax, Ymax, Zmax), N, N, N)

# Finite elements and FE function space
# Finite elements and FE function space
V = VectorElement("CG", mesh.ufl_cell(), 2)
W = FunctionSpace(mesh, V)

# Locate boundaries and initialize them to zero
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)

# functions to identify different boundaries
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], Xmin) and on_boundary

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], Xmax) and on_boundary

class Here(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], Ymin) and on_boundary

class There(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], Ymax) and on_boundary

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], Zmin) and on_boundary

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], Zmax) and on_boundary

# naming boundaries
top = Top()
bottom = Bottom()
right = Right()
left = Left()
here = Here()
there = There()

# flagging boundaries
right.mark(boundaries, 1)
left.mark(boundaries, 2)
here.mark(boundaries, 3)
there.mark(boundaries, 4)
top.mark(boundaries, 5)
bottom.mark(boundaries, 6)

dsn = ds(subdomain_data=boundaries)

# Dirichlet boundary conditions
w_top    = Constant((0.0, 0.0, 0.0))
w_right  = Constant((0.0, 0.0, 0.0))
w_here   = Constant((0.0, 0.0, 0.0))
w_there  = Constant((0.0, 0.0, 0.0))
w_bottom = Constant((0.0, 0.0, 0.0))
w_top    = Constant((0.0, 0.0, 0.0))

bc_left   = DirichletBC(W, w_top,    Left())
bc_right  = DirichletBC(W, w_right,  Right())
bc_here   = DirichletBC(W, w_here,   Here())
bc_there  = DirichletBC(W, w_there,  There())
bc_top    = DirichletBC(W, w_top,    Top())
bc_bottom = DirichletBC(W, w_bottom, Bottom())

# Collecting boundaries
bc = [bc_left, bc_right, bc_here, bc_there, bc_bottom, bc_top]

# Define constitutive relations
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
    return as_tensor( mp.C[i,j,k,l] * epsilon(u)[k,l] , (i,j))

# Define variational problem: Remember not to put plural TrialFunctions and TestFunctions
u = TrialFunction(W)
v = TestFunction(W)

# Body forces: source terms
force = Constant((0, 0, -mp.rho*9.8))
# Traction / for boundary integrals
traction_left   = Constant((0, 0, 0))
traction_right  = Constant((0, 0, 0))
traction_here   = Constant((0, 0, 0))
traction_there  = Constant((0, 0, 0))
traction_top    = Constant((0, 0, 0))
traction_bottom = Constant((0, 0, 0))

# bilinear and linear forms for the elastostatic problem
a = inner(nabla_grad(v), sigma(u)) * dx
L = dot(force, v) * dx \
+ dot(traction_left,   v) * dsn(1) \
+ dot(traction_right,  v) * dsn(2) \
+ dot(traction_here,   v) * dsn(3) \
+ dot(traction_there,  v) * dsn(4) \
+ dot(traction_top,    v) * dsn(5) \
+ dot(traction_bottom, v) * dsn(6)

# Compute FE solution
u = Function(W)
solve(a == L, u, bc)

print('min/max u  :', u.vector().min(), u.vector().max())
Elastic_energy = assemble(0.5*inner(sigma(u), epsilon(u))*dx)
print('Elastic energy  :', Elastic_energy)

u.rename('u', 'u')
File('displacement.pvd') << u
