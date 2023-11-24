"""
This is FEM solver written in FEniCs-python to solve static piezoelectric problem.
The piezoelectric materials are thought to be governed by coupling between elastic and electric problems.

How to use this script, provided FEniCs is installed:
python3 cube.py -test_number <number between 1 to 5> 

Notes:

1) This solver is specific to GENEX project task. 

2) A way of implementation of tensors is taken from the following work:
Thermodynamically consistent derivation and computation of electro-thermo-mechanical systems for solid bodies
by B. E. Abali and F. A. Reich: Transformation from Voigt to Tensor notations.

3) A way of applying poitwise boundary conditions is taken from this discussion:
https://fenicsproject.discourse.group/t/boundary-conditions-on-edges-or-nodes/5000/5    

4) https://cnam.hal.science/hal-03500725/document
"""

from __future__ import print_function
from fenics import *
import ufl as ufl
import numpy as np
# modules created locally
import voigt_to_tensor as vt
import mechanical_properties as mp
import electrical_properties as ep
import argparse

#parameters["form_compiler"]["precision"] = 100

# get runtime arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-deg', type=int, help='Enter degree', default=1)
parser.add_argument('-test_number', type=int, help='Enter test number between 1 to 5', default=1)
args = parser.parse_args()
deg = args.deg
test_no = args.test_number

# domain size
Xmin = Ymin = Zmin = 0.0
Xmax = Ymax = Zmax = 1.0

# No of points in each direction. This unusual choice is motivated by GENEX project task.
N = 1

# Variable for indices offered by dolfin for tensor algebra
i, j, k, l = ufl.indices(4)

# Create mesh and define function space
mesh = BoxMesh(Point(Xmin, Ymin, Zmin), Point(Xmax, Ymax, Zmax), N, N, N)

# Finite elements and FE function space
degree = 1
VE = VectorElement("CG", mesh.ufl_cell(), degree=deg)
PE = FiniteElement("CG", mesh.ufl_cell(), degree=deg)

# For system of PDEs
V = FunctionSpace(mesh, VE)
P = FunctionSpace(mesh, PE)
W = FunctionSpace(mesh, VE*PE)

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

# functions to identify corner nodes
class point1(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0, DOLFIN_EPS) and near(x[1], 0, DOLFIN_EPS) and near(x[2], 0, DOLFIN_EPS)

class point2(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1, DOLFIN_EPS) and near(x[1], 0, DOLFIN_EPS) and near(x[2], 0, DOLFIN_EPS)

class point3(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1, DOLFIN_EPS) and near(x[1], 1, DOLFIN_EPS) and near(x[2], 0, DOLFIN_EPS)

class point4(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0, DOLFIN_EPS) and near(x[1], 1, DOLFIN_EPS) and near(x[2], 0, DOLFIN_EPS)

class point5(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0, DOLFIN_EPS) and near(x[1], 0, DOLFIN_EPS) and near(x[2], 1, DOLFIN_EPS)

class point6(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1, DOLFIN_EPS) and near(x[1], 0, DOLFIN_EPS) and near(x[2], 1, DOLFIN_EPS)

class point7(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1, DOLFIN_EPS) and near(x[1], 1, DOLFIN_EPS) and near(x[2], 1, DOLFIN_EPS)

class point8(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0, DOLFIN_EPS) and near(x[1], 1, DOLFIN_EPS) and near(x[2], 1, DOLFIN_EPS)

# flag corner nodes
p1 = point1()
p2 = point2()
p3 = point3()
p4 = point4()
p5 = point5()
p6 = point6()
p7 = point7()
p8 = point8()

dsn = ds(subdomain_data=boundaries)

# Mark facets
n = FacetNormal(mesh)

# Dirichlet boundary conditions applied to faces
w_top    = Constant((0.0, 0.0, 0.0, 0.0))
w_right  = Constant((0.0, 0.0, 0.0, 0.0))
w_here   = Constant((0.0, 0.0, 0.0, 0.0))
w_there  = Constant((0.0, 0.0, 0.0, 0.0))
w_bottom = Constant((0.0, 0.0, 0.0, 0.0))
w_top    = Constant((0.0, 0.0, 0.0, 0.0))

bc_left   = DirichletBC(W, w_top,    Left())
bc_right  = DirichletBC(W, w_right,  Right())
bc_here   = DirichletBC(W, w_here,   Here())
bc_there  = DirichletBC(W, w_there,  There())
bc_top    = DirichletBC(W, w_top,    Top())
bc_bottom = DirichletBC(W, w_bottom, Bottom())

# Dirichlet boundary conditions applied to nodes
bc1      = DirichletBC(W.sub(0), Constant((0.0, 0.0, 0.0)), p1, method="pointwise")
bc22     = DirichletBC(W.split()[0].sub(1), Constant(0.0), p2, method="pointwise")
bc23     = DirichletBC(W.split()[0].sub(2), Constant(0.0), p2, method="pointwise")
bc43     = DirichletBC(W.split()[0].sub(0), Constant(0.0), p4, method="pointwise")

bc1p     = DirichletBC(W.split()[1], Constant(1.0), p1, method="pointwise")
bc2p     = DirichletBC(W.split()[1], Constant(1.0), p2, method="pointwise")
bc3p     = DirichletBC(W.split()[1], Constant(1.0), p3, method="pointwise")
bc4p     = DirichletBC(W.split()[1], Constant(1.0), p4, method="pointwise")

bc5p     = DirichletBC(W.split()[1], Constant(0.0), p5, method="pointwise")
bc6p     = DirichletBC(W.split()[1], Constant(0.0), p6, method="pointwise")
bc7p     = DirichletBC(W.split()[1], Constant(0.0), p7, method="pointwise")
bc8p     = DirichletBC(W.split()[1], Constant(0.0), p8, method="pointwise")

# Collecting boundaries depeding upon test case
if test_no == 1:
    bc1      = DirichletBC(W.sub(0), Constant((0.0, 0.0, 0.0)), p1, method="pointwise")
    bc22     = DirichletBC(W.split()[0].sub(1), Constant(0.0), p2, method="pointwise")
    bc23     = DirichletBC(W.split()[0].sub(2), Constant(0.0), p2, method="pointwise")
    bc43     = DirichletBC(W.split()[0].sub(2), Constant(0.0), p4, method="pointwise")

    bc1p     = DirichletBC(W.split()[1], Constant(1.0), p1, method="pointwise")
    bc2p     = DirichletBC(W.split()[1], Constant(1.0), p2, method="pointwise")
    bc3p     = DirichletBC(W.split()[1], Constant(1.0), p3, method="pointwise")
    bc4p     = DirichletBC(W.split()[1], Constant(1.0), p4, method="pointwise")
    
    bc5p     = DirichletBC(W.split()[1], Constant(0.0), p5, method="pointwise")
    bc6p     = DirichletBC(W.split()[1], Constant(0.0), p6, method="pointwise")
    bc7p     = DirichletBC(W.split()[1], Constant(0.0), p7, method="pointwise")
    bc8p     = DirichletBC(W.split()[1], Constant(0.0), p8, method="pointwise")
    
    bc = [bc1, bc22, bc23, bc43, bc1p ,bc2p, bc3p, bc4p, bc5p, bc6p, bc7p, bc8p]

elif test_no == 2:
    bc1      = DirichletBC(W.sub(0), Constant((0.0, 0.0, 0.0)), p1, method="pointwise")
    bc22     = DirichletBC(W.split()[0].sub(1), Constant(0.0), p2, method="pointwise")
    bc23     = DirichletBC(W.split()[0].sub(2), Constant(0.0), p2, method="pointwise")
    bc43     = DirichletBC(W.split()[0].sub(2), Constant(0.0), p4, method="pointwise")
    
    bc1p     = DirichletBC(W.split()[1], Constant(0.0), p1, method="pointwise")
    bc2p     = DirichletBC(W.split()[1], Constant(0.0), p2, method="pointwise")
    bc3p     = DirichletBC(W.split()[1], Constant(0.0), p3, method="pointwise")
    bc4p     = DirichletBC(W.split()[1], Constant(0.0), p4, method="pointwise")

    bc5p     = DirichletBC(W.split()[1], Constant(1.0), p5, method="pointwise")
    bc6p     = DirichletBC(W.split()[1], Constant(1.0), p6, method="pointwise")
    bc7p     = DirichletBC(W.split()[1], Constant(1.0), p7, method="pointwise")
    bc8p     = DirichletBC(W.split()[1], Constant(1.0), p8, method="pointwise")
    
    bc = [bc1, bc22, bc23, bc43, bc1p ,bc2p, bc3p, bc4p, bc5p, bc6p, bc7p, bc8p]

elif test_no == 3:
    bc1      = DirichletBC(W.sub(0), Constant((0.0, 0.0, 0.0)), p1, method="pointwise")
    bc22     = DirichletBC(W.split()[0].sub(1), Constant(0.0), p2, method="pointwise")
    bc23     = DirichletBC(W.split()[0].sub(2), Constant(0.0), p2, method="pointwise")
    bc43     = DirichletBC(W.split()[0].sub(2), Constant(0.0), p4, method="pointwise")

    bc2p     = DirichletBC(W.split()[1], Constant(0.0), p2, method="pointwise")
    bc3p     = DirichletBC(W.split()[1], Constant(0.0), p3, method="pointwise")
    bc6p     = DirichletBC(W.split()[1], Constant(0.0), p6, method="pointwise")
    bc7p     = DirichletBC(W.split()[1], Constant(0.0), p7, method="pointwise")

    bc1p     = DirichletBC(W.split()[1], Constant(1.0), p1, method="pointwise")
    bc4p     = DirichletBC(W.split()[1], Constant(1.0), p4, method="pointwise")
    bc5p     = DirichletBC(W.split()[1], Constant(1.0), p5, method="pointwise")
    bc8p     = DirichletBC(W.split()[1], Constant(1.0), p8, method="pointwise")
    
    bc = [bc1, bc22, bc23, bc43, bc1p ,bc2p, bc3p, bc4p, bc5p, bc6p, bc7p, bc8p]

elif test_no == 4:
    bc13     = DirichletBC(W.split()[0].sub(2), Constant(0.0), p1, method="pointwise")
    bc23     = DirichletBC(W.split()[0].sub(2), Constant(0.0), p2, method="pointwise")
    bc33     = DirichletBC(W.split()[0].sub(2), Constant(0.0), p3, method="pointwise")
    bc43     = DirichletBC(W.split()[0].sub(2), Constant(0.0), p4, method="pointwise")

    bc12     = DirichletBC(W.split()[0].sub(1), Constant(0.0), p1, method="pointwise")
    bc22     = DirichletBC(W.split()[0].sub(1), Constant(0.0), p2, method="pointwise")
    bc52     = DirichletBC(W.split()[0].sub(1), Constant(0.0), p5, method="pointwise")
    bc62     = DirichletBC(W.split()[0].sub(1), Constant(0.0), p6, method="pointwise")

    bc11     = DirichletBC(W.split()[0].sub(0), Constant(0.0), p1, method="pointwise")
    bc41     = DirichletBC(W.split()[0].sub(0), Constant(0.0), p4, method="pointwise")
    bc51     = DirichletBC(W.split()[0].sub(0), Constant(0.0), p5, method="pointwise")
    bc81     = DirichletBC(W.split()[0].sub(0), Constant(0.0), p8, method="pointwise")

    bc1p     = DirichletBC(W.split()[1], Constant(0.0), p1, method="pointwise")
    bc2p     = DirichletBC(W.split()[1], Constant(0.0), p2, method="pointwise")
    bc3p     = DirichletBC(W.split()[1], Constant(0.0), p3, method="pointwise")
    bc4p     = DirichletBC(W.split()[1], Constant(0.0), p4, method="pointwise")

    bc = [bc13, bc23, bc33, bc43, bc12, bc22, bc52, bc62, bc11, bc41, bc51, bc81, bc1p ,bc2p, bc3p, bc4p]

elif test_no == 5:
    bc13     = DirichletBC(W.split()[0].sub(2), Constant(0.0), p1, method="pointwise")
    bc23     = DirichletBC(W.split()[0].sub(2), Constant(0.0), p2, method="pointwise")
    bc33     = DirichletBC(W.split()[0].sub(2), Constant(0.0), p3, method="pointwise")
    bc43     = DirichletBC(W.split()[0].sub(2), Constant(0.0), p4, method="pointwise")

    bc12     = DirichletBC(W.split()[0].sub(1), Constant(0.0), p1, method="pointwise")
    bc22     = DirichletBC(W.split()[0].sub(1), Constant(0.0), p2, method="pointwise")
    bc52     = DirichletBC(W.split()[0].sub(1), Constant(0.0), p5, method="pointwise")
    bc62     = DirichletBC(W.split()[0].sub(1), Constant(0.0), p6, method="pointwise")

    bc11     = DirichletBC(W.split()[0].sub(0), Constant(0.0), p1, method="pointwise")
    bc41     = DirichletBC(W.split()[0].sub(0), Constant(0.0), p4, method="pointwise")
    bc51     = DirichletBC(W.split()[0].sub(0), Constant(0.0), p5, method="pointwise")
    bc81     = DirichletBC(W.split()[0].sub(0), Constant(0.0), p8, method="pointwise")

    bc1p     = DirichletBC(W.split()[1], Constant(0.0), p1, method="pointwise")
    bc2p     = DirichletBC(W.split()[1], Constant(0.0), p2, method="pointwise")
    bc3p     = DirichletBC(W.split()[1], Constant(0.0), p3, method="pointwise")
    bc4p     = DirichletBC(W.split()[1], Constant(0.0), p4, method="pointwise")

    bc = [bc13, bc23, bc33, bc43, bc12, bc22, bc52, bc62, bc11, bc41, bc51, bc81, bc1p ,bc2p, bc3p, bc4p]
    
def matrix2vector(M):
    return np.array([M[0,0], M[1,1], M[2,2], M[0,1], M[0,2], M[1,2]])

def vector2matrix(v):
    return np.array([[v[0], v[3], v[4]], [v[3], v[1], v[5]], [v[4], v[5], v[2]]])
            
# Define constitutive relations
# Mechanical Strain and Stress tensors
def epsilon(u):
    return as_vector(matrix2vector(0.5*(nabla_grad(u) + nabla_grad(u).T)))

def sigma_u(u):
    #return as_tensor( mp.C[i,j,k,l] * epsilon(u)[k,l] , (i,j))
    return as_tensor( vector2matrix( as_vector( mp.elasticity_tensor[i,j] * epsilon(u)[j], (i) )) )

# Piezoelectric/elecromechonical tensor
def sigma_p(phi):
    #return as_tensor( ep.Ephi[j,k,i] * grad(phi)[i], (j,k))
    return as_tensor( vector2matrix( as_vector( ep.piezoelectric_e_tensor.T[i,j] * grad(phi)[j], (i)) ))

def edisp_u(u):
    #return as_tensor( ep.Ephi[i,j,k] * epsilon(u)[j,k], (i))
    return as_vector( ep.piezoelectric_e_tensor[i,j] * epsilon(u)[j], (i)) 

# Electrostatic vector
def edisp_p(phi):
    return ep.dielectric_tensor * grad(phi)

# Define variational problem
(u, phi) = TrialFunctions(W)
(v, q)   = TestFunctions(W)

# Body forces: source terms
force = Constant((0, 0, 0))
# Charge: source term
charge  = Constant(0)

# Traction for boundary integrals
traction_left   = Constant((0, 0, 0))
traction_right  = Constant((0, 0, 0))
traction_here   = Constant((0, 0, 0))
traction_there  = Constant((0, 0, 0))
traction_top    = Constant((0, 0, 0))
traction_bottom = Constant((0, 0, 0))

if test_no == 4: traction_top    = Constant((0, 0, 1)) # be careful about signs
if test_no == 5: traction_right  = Constant((1, 0, 0))

# Voltage for boundary integrals
gradphi_left   = Constant(0.0)
gradphi_right  = Constant(0.0)
gradphi_here   = Constant(0.0)
gradphi_there  = Constant(0.0)
gradphi_top    = Constant(0.0)
gradphi_bottom = Constant(0.0)

# bilinear and linear forms for the piezoelectric static problem
a =  inner(nabla_grad(v), sigma_u(u)   ) * dx \
+ inner(nabla_grad(v), sigma_p(phi) ) * dx   \
- inner(grad(q)      , edisp_u(u)   ) * dx   \
+ inner(grad(q)      , edisp_p(phi) ) * dx

L = dot(force, v) * dx + q * charge * dx \
+ dot(traction_left,   v) * dsn(2) \
+ dot(traction_right,  v) * dsn(1) \
+ dot(traction_here,   v) * dsn(3) \
+ dot(traction_there,  v) * dsn(4) \
+ dot(traction_top,    v) * dsn(5) \
+ dot(traction_bottom, v) * dsn(6) \
+ q * gradphi_left   * dsn(2) \
+ q * gradphi_right  * dsn(1) \
+ q * gradphi_here   * dsn(3) \
+ q * gradphi_there  * dsn(4) \
+ q * gradphi_top    * dsn(5) \
+ q * gradphi_bottom * dsn(6)

# Compute FE solution
U = Function(W)
solve(a == L, U, bc)

# Get sub-functions
u, phi = U.split()

# Printitng some quantities
print('min/max u  :', u.vector().min(), u.vector().max())      
print('min/max phi:', phi.vector().min(), phi.vector().max())

# Rename functions: This helps to get post-processing files with proper names for data.
# It seems second argument has no effect at all.
u.rename('u', 'u')
phi.rename('phi', 'phi')

# Save solution and some quantities to files in VTK format
File('solution/Displacement.pvd') << u
File('solution/Potential.pvd') << phi

if deg > 1:
    VFS = VectorFunctionSpace(mesh, "CG", deg-1)
    TVS = TensorFunctionSpace(mesh, "CG", deg-1)    
else:
    VFS = VectorFunctionSpace(mesh, "DG", 0)
    TVS = TensorFunctionSpace(mesh, "DG", 0)

Ed = Function(VFS, name="Electrical_Displacement")
#q_vec = as_vector( ep.Ephi[i,k,l] * nabla_grad(u)[k,l], (i)) - ep.dielectric_tensor * grad(phi)
q_vec = as_vector( ep.piezoelectric_e_tensor[i,j] * epsilon(u)[j], (i) ) - ep.dielectric_tensor * grad(phi)
Ed.assign(project(q_vec, VFS))
File('solution/Electrical_Displacement.pvd') << Ed

Ep = Function(VFS, name="Electric_Field")
q_vec = - grad(phi)
Ep.assign(project(q_vec, VFS))
File('solution/Electrical_Field.pvd') << Ep

Ms = Function(TVS, name="Mechanical_Strain")
#Ms.assign(project(0.5*(nabla_grad(u) + nabla_grad(u).T), TVS))
Ms.assign(project(as_tensor( vector2matrix( as_vector( epsilon(u)[i], (i) )) ) , TVS))
File('solution/Mechanical_Strain.pvd') << Ms

Ms = Function(TVS, name="Mechanical_Stress")
Ms.assign(project(as_tensor( vector2matrix( as_vector( mp.elasticity_tensor[i,j] * epsilon(u)[j], (i) )) ), TVS) )
File('solution/Mechanical_Stress.pvd') << Ms
