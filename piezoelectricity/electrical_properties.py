import numpy as np
import voigt_to_tensor as vt
from dolfin import *

D_1 = 1.505e-08
D_2 = 1.505e-08
D_3 = 1.301e-08

# dielectric tensor [C.m-1.V-1]
dielectric_tensor = as_matrix([ [D_1, 0.0, 0.0], [0.0, D_2, 0.0], [0.0, 0.0, D_3] ])

# piezoelectric properties
e1_11 = 0.0
e1_22 = 0.0
e1_33 = 0.0

e1_12 = 0.0
e1_13 = 741.0E-12
e1_23 = 0.0

e2_11 = 0.0
e2_22 = 0.0
e2_33 = 0.0

e2_12 = 0.0
e2_13 = 0.0
e2_23 = 741.0E-12

e3_11 = -274.0E-12
e3_12 = -274.0E-12
e3_13 =  593.0E-12

e3_12 = 0.0
e3_13 = 0.0
e3_23 = 0.0

# piezoelectric coupling tensor
piezoelectric_tensor = as_matrix([ [e1_11, e1_22, e1_33, e1_12, e1_13, e1_23], \
                                   [e2_11, e2_22, e2_33, e2_12, e2_13, e2_23], \
                                   [e3_11, e3_12, e3_13, e3_12, e3_13, e3_23] ])

Ephi = vt.VoigtToTensorRank3(A11=e1_11,A12=e1_22,A13=e1_33,A14=e1_12,A15=e1_13,A16=e1_23, \
A21=e2_11,A22=e2_22,A23=e2_33,A24=e2_12,A25=e2_13,A26=e2_23, \
A31=e3_11,A32=e3_12,A33=e3_13,A34=e3_12,A35=e3_13,A36=e3_23)

