from dolfin import *
import voigt_to_tensor as vt

# Material density kg/m^3
rho = 7500

E_1 = 6.061e10 
E_2 = 6.061e10
E_3 = 4.831e10

nu_12 = 0.289
nu_13 = 0.512
nu_23 = 0.512

G_12 = 2.35E10
G_13 = 2.30E10
G_23 = 2.30e10

# https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/usb/default.htm?startat=pt05ch17s02abm02.html
# https://mooseframework.inl.gov/source/materials/ComputeElasticityTensor.html
delta = 1.0 - nu_12*nu_12 - nu_13*nu_13 - nu_23*nu_23 - 2.0*nu_12*nu_13*nu_23

e11 = (E_1 - nu_23*nu_23) / delta
e12 = E_1 * (nu_12 + nu_13 * nu_23) / delta
e13 = E_1 * (nu_13 + nu_12 * nu_23) / delta

e22 = (E_2 - nu_13*nu_13) / delta
e23 = E_2 * (nu_23 + nu_12 * nu_13) / delta

e33 = (E_3 - nu_12*nu_12) / delta

e44 = G_12
e55 = G_13
e66 = G_23

# Elasticity matrix: This should be 4th order tensor in general.
elasticity_tensor = as_matrix([ [e11, e12, e13, 0.0, 0.0, 0.0], 
                                [e12, e22, e23, 0.0, 0.0, 0.0], 
                                [e13, e23, e33, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, e44, 0.0, 0.0], 
                                [0.0, 0.0, 0.0, 0.0, e55, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, e66]  ])

C = vt.VoigtToTensorRank4(A11=e11,A12=e12,A13=e13,A22=e22,A23=e13,A33=e33,A44=e44,A55=e55,A66=e66)
