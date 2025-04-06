import numpy as np

def the_function(x, n):
    # Compute the sum over differences with varying exponents.
    # exponents: [2, 4, 6, ..., 2*(n-1)]
    exponents = 2 * np.arange(1, n)
    diff = x[:-1] - x[1:]
    return x[0]**2 + np.sum(np.power(diff, exponents))

def the_function_gradient(x, n):
    grad = np.zeros(n, dtype=x.dtype)
    # The first term from x[0]**2
    grad[0] = 2 * x[0]
    # exponents for the terms: 2*(i+1)-1 for i=0,..., n-2 => [1, 3, 5, ...]
    exponents = 2 * np.arange(1, n) - 1
    coeff = 2 * np.arange(1, n)
    # Compute the term for each difference.
    diff = x[:-1] - x[1:]
    terms = coeff * np.power(diff, exponents)
    # Add/subtract the terms to the corresponding components.
    grad[:-1] += terms
    grad[1:] -= terms
    return grad

def the_function_hessian(x, n):
    H = np.zeros((n, n), dtype=x.dtype)
    H[0, 0] = 2
    # exponents for the Hessian terms: 2*(i+1)-2 for i=0,..., n-2 => [0, 2, 4, ...]
    exponents = 2 * np.arange(1, n) - 2
    coeff = 2 * np.arange(1, n) * (2 * np.arange(1, n) - 1)
    diff = x[:-1] - x[1:]
    factors = coeff * np.power(diff, exponents)
    
    indices = np.arange(n-1)
    H[indices, indices] += factors
    H[indices, indices+1] -= factors
    H[indices+1, indices] -= factors
    H[indices+1, indices+1] += factors
    return H
