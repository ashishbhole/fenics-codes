from dolfin import *
import numpy as np
import voigt_to_tensor as vt

# Material density kg/m^3
rho = 7500
E_1 = 6.061e10 ; E_2 = 6.061e10 ; E_3 = 4.831e10
nu_12 = 0.289 ; nu_13 = 0.512 ; nu_23 = 0.512
G_12 = 2.35E10 ; G_13 = 2.30E10 ; G_23 = 2.30e10

compiliance_matrix = np.array([ [1/E_1, -nu_12/E_1, -nu_13/E_1, 0.0, 0.0, 0.0],
                                [-nu_12/E_1, 1/E_2, -nu_23/E_2, 0.0, 0.0, 0.0],
                                [-nu_13/E_1, -nu_23/E_2, 1/E_3, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1/G_12, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1/G_23, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1/G_13]  ])
elasticity_tensor = np.linalg.inv(compiliance_matrix)

C11 = elasticity_tensor[0][0]
C22 = elasticity_tensor[1][1]
C33 = elasticity_tensor[2][2]

C12 = elasticity_tensor[0][1]
C13 = elasticity_tensor[0][2]
C23 = elasticity_tensor[1][2]

C21 = elasticity_tensor[1][0]
C31 = elasticity_tensor[2][0]
C32 = elasticity_tensor[2][1]

C44 = elasticity_tensor[3][3]
C55 = elasticity_tensor[4][4]
C66 = elasticity_tensor[5][5]

C = vt.VoigtToTensorRank4(A11=C11,A12=C12,A13=C13,A22=C22,A23=C23,A33=C33,A44=C44,A55=C55,A66=C66)

'''
# https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/usb/default.htm?startat=pt05ch17s02abm02.html
# https://mooseframework.inl.gov/source/materials/ComputeElasticityTensor.html
# Mechanics of Composite Materials 2nd ed 1999 Robert Jones (Taylor & Franics)

delta = (1.0 - nu_12*nu_12 - nu_13*nu_13 - nu_23*nu_23 - 2.0*nu_12*nu_13*nu_23) / (E_1*E_2*E_3)

e11 = (1.0 - nu_23*nu_23) / (E_2 * E_3 * delta)
e22 = (1.0 - nu_13*nu_13) / (E_1 * E_3 * delta)
e33 = (1.0 - nu_12*nu_12) / (E_1 * E_2 * delta)

e12 = (nu_12 + nu_13 * nu_23) / (E_2 * E_3 * delta)
e13 = (nu_13 + nu_12 * nu_23) / (E_1 * E_2 * delta)

e23 = (nu_23 + nu_12 * nu_13) / (E_1 * E_2 * delta)

e44 = G_23
e55 = G_13
e66 = G_12

# Elasticity matrix: This should be 4th order tensor in general.
elasticity_tensor = as_matrix([ [e11, e12, e13, 0.0, 0.0, 0.0], 
                                [e12, e22, e23, 0.0, 0.0, 0.0], 
                                [e13, e23, e33, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, e44, 0.0, 0.0], 
                                [0.0, 0.0, 0.0, 0.0, e55, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, e66]  ])

C = vt.VoigtToTensorRank4(A11=e11,A12=e12,A13=e13,A22=e22,A23=e13,A33=e33,A44=e44,A55=e55,A66=e66)
'''
