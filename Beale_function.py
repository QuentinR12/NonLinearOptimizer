import numpy as np

def beale(x):
    x1, x2 = x
    return (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2

def beale_gradient(x):
    x1, x2 = x
    u1 = 1.5 - x1 + x1*x2
    u2 = 2.25 - x1 + x1*x2**2
    u3 = 2.625 - x1 + x1*x2**3
    grad_x1 = 2*u1*(-1+x2) + 2*u2*(-1+x2**2) + 2*u3*(-1+x2**3)
    grad_x2 = 2*u1*x1 + 2*u2*(2*x1*x2) + 2*u3*(3*x1*x2**2)
    return np.array([grad_x1, grad_x2])

def beale_hessian(x):
    x1, x2 = x
    u1 = 1.5 - x1 + x1*x2
    u2 = 2.25 - x1 + x1*x2**2
    u3 = 2.625 - x1 + x1*x2**3
    u1_x1 = -1 + x2;    u1_x2 = x1
    u2_x1 = -1 + x2**2; u2_x2 = 2*x1*x2
    u3_x1 = -1 + x2**3; u3_x2 = 3*x1*x2**2
    u1_x1x2 = 1;       u2_x1x2 = 2*x2;   u2_x2x2 = 2*x1
    u3_x1x2 = 3*x2**2; u3_x2x2 = 6*x1*x2
    f_x1x1 = 2*(u1_x1**2 + u2_x1**2 + u3_x1**2)
    f_x1x2 = 2*(u1_x1*u1_x2 + u2_x1*u2_x2 + u3_x1*u3_x2) + 2*(u1*u1_x1x2 + u2*u2_x1x2 + u3*u3_x1x2)
    f_x2x2 = 2*(u1_x2**2 + u2_x2**2 + u3_x2**2) + 2*(u2*u2_x2x2 + u3*u3_x2x2)
    return np.array([[f_x1x1, f_x1x2], [f_x1x2, f_x2x2]])