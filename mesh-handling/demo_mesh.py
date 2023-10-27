from dolfin import *
# to list object attributes
import inspect as ins
# for nice prints
from pprint import pprint

# Create mesh and define function space
mesh = UnitSquareMesh(10,10)

# nicely prints all mesh attributes
with open('mesh_attributes.txt','w') as f:
  pprint(dir(mesh), f)

print("Mesh Type", mesh.type)

# ufl related
print("(ufl) Cell type : ", mesh.ufl_cell())
print("(ufl) domain type : ", mesh.ufl_domain())
print("(ufl) coordinate element : ", mesh.ufl_coordinate_element())
print("(ufl) id : ", mesh.ufl_id())

print("(ufl) Cell name : ", mesh.cell_name())

#print("Mesh topology", mesh.topology())

print("Number of vertices : ", mesh.num_vertices())
print("Number of cells : ", mesh.num_cells())
print("Number of edges : ", mesh.num_edges())
print("Number of faces : ", mesh.num_faces())
print("Number of facets : ", mesh.num_facets())
print("Number of entities : ", mesh.num_entities(0))

print("If ordered : ", mesh.ordered())

print("Minimum and maximum cell diameters: ", mesh.hmin(), mesh.hmax())
print("Minimum and maximum cell inradius: ", mesh.rmin(), mesh.rmax())

coords = mesh.coordinates()

# let's iterate over vertices
for v in vertices(mesh):
    print('vertex index and coordinates : ', v.index(), coords[v.index()])

# 'long' version
for c in cells(mesh):
  print("cell", c.index(), "has edges :", c.entities(1))

# Save solution in VTK format
file = File("mesh.pvd")
file << mesh

#pprint(ins.getmembers(mesh)) #this gives the keys
