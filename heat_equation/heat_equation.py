# Based on https://fenicsproject.org/pub/tutorial/html/._ftut1006.html
from fenics import *
import time

nu = 1e-3          # diffusivity
adv_x = adv_y = 0.1# advection speed
T = 2.0            # final time
num_steps = 50     # number of time steps
dt = T / num_steps # time step size

# Create mesh and define function space
nx = ny = 30
deg = 1
mesh = RectangleMesh(Point(-2, -2), Point(2, 2), nx, ny)
V = FunctionSpace(mesh, 'P', deg)

# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0), boundary)

# Define initial value
#u_0 = Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))',
#                 degree=2, a=5)
u_0 = Expression('0.0', degree=2)
u_n = interpolate(u_0, V)

speed = as_vector([adv_x, adv_y])

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

f = Expression('(n <= 1) ? (a*exp(-b*pow(x[0], 2) - b*pow(x[1], 2))) : 0.0 ', n=0, a=1e10, b=32, degree=deg+1)
# do not interpolate f on V, otherwise it does not work in time loop. why?

# Variational form
F = u*v*dx + dt*v*dot(speed, grad(u))*dx + dt*nu*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Create VTK file for saving solution
vtkfile = File('numerical_solution/solution.pvd')

# Time-stepping
u = Function(V)
t = 0
u.rename('u','u')
vtkfile << (u, t)
for n in range(num_steps):
    # Update current time
    t += dt

    # Update source declared as Expression
    f.n = n

    # Compute solution
    solve(a == L, u, bc)

    # Save to file and plot solution
    vtkfile << (u, t)

    # Update previous solution
    u_n.assign(u)
