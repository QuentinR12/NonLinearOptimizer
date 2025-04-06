import numpy as np

def Rosenbrock(x):
    n = x.size
    f = 0.0
    for i in range(1, n):
        xi_prev = x[i-1]
        # Use multiplication instead of **2 for clarity and speed
        term = 100 * (x[i] - xi_prev * xi_prev)**2 + (1 - xi_prev)**2
        f += term
    return f

def Rosenbrock_gradient(x):
    n = x.size
    grad = np.empty(n, dtype=x.dtype)
    # First component
    grad[0] = -400 * x[0] * x[1] + 400 * x[0] * x[0] * x[0] + 2 * (x[0] - 1)
    # Middle components
    for i in range(1, n - 1):
        grad[i] = (-400 * x[i] * x[i+1] + 400 * x[i] * x[i] * x[i] + 2 * (x[i] - 1)) \
                  + (200 * x[i] - 200 * x[i-1] * x[i-1])
    # Last component
    grad[n - 1] = 200 * x[n - 1] - 200 * x[n - 2] * x[n - 2]
    return grad

def Rosenbrock_hessian(x):
    n = x.size
    hess = np.zeros((n, n), dtype=x.dtype)
    for i in range(n - 1):
        # Diagonal term for the i-th row
        hess[i, i] = -400 * x[i+1] + 1200 * x[i] * x[i] + 2
        # Off-diagonal term: derivative with respect to x[i+1]
        hess[i, i+1] = -400 * x[i]
        # Symmetric entry
        hess[i+1, i] = -400 * x[i]
        # Diagonal term for the (i+1)-th row
        hess[i+1, i+1] = 200
    return hess
