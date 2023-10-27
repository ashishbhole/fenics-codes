"""
Adapted from: FEniCS tutorial demo program: Linear electostatic problem.

Problem: Electrostatics
  -div(nable(phi)) = q
"""
from __future__ import print_function
from fenics import *
import numpy as np

# No of points in each direction
N = 5

# domain size
Xmin = Ymin = Zmin = 0.0
Xmax = Ymax = Zmax = 1.0

# dielectric tensor [C.m-1.V-1]
D_1 = 1.505e-08
D_2 = 1.505e-08
D_3 = 1.301e-08
dielectric_tensor = as_matrix([ [D_1, 0.0, 0.0], [0.0, D_2, 0.0], [0.0, 0.0, D_3] ])

# Create mesh and define function space
mesh = BoxMesh(Point(Xmin, Ymin, Zmin), Point(Xmax, Ymax, Zmax), N, N, N)

# Finite elements and FE function space
P = FunctionSpace(mesh, "CG", 1)

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
p_top    = Constant(0.0)
p_right  = Constant(0.0)
p_here   = Constant(0.0)
p_there  = Constant(0.0)
p_bottom = Constant(1.0)
p_top    = Constant(0.0)

bc_left   = DirichletBC(P, p_top,    Left())
bc_right  = DirichletBC(P, p_right,  Right())
bc_here   = DirichletBC(P, p_here,   Here())
bc_there  = DirichletBC(P, p_there,  There())
bc_top    = DirichletBC(P, p_top,    Top())
bc_bottom = DirichletBC(P, p_bottom, Bottom())

# Collecting boundaries
bc = [bc_left, bc_right, bc_here, bc_there, bc_bottom, bc_top]

# Define variational problem
phi  = TrialFunction(P)
phi_ = TestFunction(P)

# Charge: source term
charge  = Constant((0.0))
# Voltage / for boundary integrals
gradphi_left   = Constant(0.0)
gradphi_right  = Constant(0.0)
gradphi_here   = Constant(0.0)
gradphi_there  = Constant(0.0)
gradphi_top    = Constant(0.0)
gradphi_bottom = Constant(1.0)

a = dot( grad(phi_), dielectric_tensor*grad(phi) ) * dx
L = phi_ * charge * dx       \
+ phi_ * gradphi_left   * dsn(1) \
+ phi_ * gradphi_right  * dsn(2) \
+ phi_ * gradphi_here   * dsn(3) \
+ phi_ * gradphi_there  * dsn(4) \
+ phi_ * gradphi_top    * dsn(5) \
+ phi_ * gradphi_bottom * dsn(6)

phi = Function(P)
solve(a == L, phi, bc)
electric_field = project(grad(phi))

print('min/max phi:', phi.vector().min(), phi.vector().max())

phi.rename('phi', 'phi')
File('potential.pvd') << phi
File('electric_field.pvd') << electric_field
