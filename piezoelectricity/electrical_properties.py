import numpy as np
import voigt_to_tensor as vt
import mechanical_properties as mp
from dolfin import *

D_1 = 1.505e-08
D_2 = 1.505e-08
D_3 = 1.301e-08

# dielectric tensor [C.m-1.V-1]
dielectric_tensor = as_matrix([ [D_1, 0.0, 0.0], [0.0, D_2, 0.0], [0.0, 0.0, D_3] ])

# piezoelectric properties
d1_11 = 0.0
d1_22 = 0.0
d1_33 = 0.0

d1_12 = 0.0
d1_13 = 741.0E-12
d1_23 = 0.0

d2_11 = 0.0
d2_22 = 0.0
d2_33 = 0.0

d2_12 = 0.0
d2_13 = 0.0
d2_23 = 741.0E-12

d3_11 = -274.0E-12
d3_22 = -274.0E-12
d3_33 =  593.0E-12

d3_12 = 0.0
d3_13 = 0.0
d3_23 = 0.0

# piezoelectric coupling tensor
piezoelectric_d_tensor = as_matrix([ [d1_11, d1_22, d1_33, d1_12, d1_13, d1_23], \
                                     [d2_11, d2_22, d2_33, d2_12, d2_13, d2_23], \
                                     [d3_11, d3_22, d3_33, d3_12, d3_13, d3_23] ])

piezoelectric_e_tensor = as_matrix(piezoelectric_d_tensor * mp.elasticity_tensor)

#Ephi = vt.VoigtToTensorRank3(A11=e1_11,A12=e1_22,A13=e1_33,A14=e1_12,A15=e1_13,A16=e1_23, \
#A21=e2_11,A22=e2_22,A23=e2_33,A24=e2_12,A25=e2_13,A26=e2_23, \
#A31=e3_11,A32=e3_22,A33=e3_33,A34=e3_12,A35=e3_13,A36=e3_23)

