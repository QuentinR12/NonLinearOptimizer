import numpy as np

def beale(x):
    x1, x2 = x[0], x[1]
    # Precompute powers of x2
    x2_2 = x2 * x2
    x2_3 = x2 * x2_2
    # Compute terms
    term1 = 1.5 - x1 + x1 * x2
    term2 = 2.25 - x1 + x1 * x2_2
    term3 = 2.625 - x1 + x1 * x2_3
    return term1**2 + term2**2 + term3**2

def beale_gradient(x):
    x1, x2 = x[0], x[1]
    # Precompute powers of x2
    x2_2 = x2 * x2
    x2_3 = x2 * x2_2
    # Compute common terms
    u1 = 1.5 - x1 + x1 * x2
    u2 = 2.25 - x1 + x1 * x2_2
    u3 = 2.625 - x1 + x1 * x2_3
    factor = 2.0
    grad_x1 = factor * (u1 * (x2 - 1) + u2 * (x2_2 - 1) + u3 * (x2_3 - 1))
    grad_x2 = factor * (u1 * x1 + u2 * (2 * x1 * x2) + u3 * (3 * x1 * x2_2))
    out = np.empty(2, dtype=np.float64)
    out[0] = grad_x1
    out[1] = grad_x2
    return out

def beale_hessian(x):
    x1, x2 = x[0], x[1]
    # Precompute powers of x2
    x2_2 = x2 * x2
    x2_3 = x2 * x2_2
    # Compute u terms
    u1 = 1.5 - x1 + x1 * x2
    u2 = 2.25 - x1 + x1 * x2_2
    u3 = 2.625 - x1 + x1 * x2_3
    # First derivatives of u's
    u1_x1 = x2 - 1
    u1_x2 = x1
    u2_x1 = x2_2 - 1
    u2_x2 = 2 * x1 * x2
    u3_x1 = x2_3 - 1
    u3_x2 = 3 * x1 * x2_2
    # Mixed partial derivatives
    u1_x1x2 = 1
    u2_x1x2 = 2 * x2
    u3_x1x2 = 3 * x2_2
    # Second derivatives with respect to x2
    u2_x2x2 = 2 * x1
    u3_x2x2 = 6 * x1 * x2
    factor = 2.0
    f_x1x1 = factor * (u1_x1 * u1_x1 + u2_x1 * u2_x1 + u3_x1 * u3_x1)
    f_x1x2 = factor * (u1_x1 * u1_x2 + u2_x1 * u2_x2 + u3_x1 * u3_x2) \
             + factor * (u1 * u1_x1x2 + u2 * u2_x1x2 + u3 * u3_x1x2)
    f_x2x2 = factor * (u1_x2 * u1_x2 + u2_x2 * u2_x2 + u3_x2 * u3_x2) \
             + factor * (u2 * u2_x2x2 + u3 * u3_x2x2)
    out = np.empty((2, 2), dtype=np.float64)
    out[0, 0] = f_x1x1
    out[0, 1] = f_x1x2
    out[1, 0] = f_x1x2
    out[1, 1] = f_x2x2
    return out
