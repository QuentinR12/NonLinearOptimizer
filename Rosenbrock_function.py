import numpy as np

def Rosenbrock(x):
    n = x.size
    return sum(100 * (x[i] - x[i-1]**2)**2 + (1 - x[i-1])**2 for i in range(1, n))

def Rosenbrock_gradient(x):
    n = x.size
    grad = np.empty(n)
    grad[0] = -400 * x[0] * x[1] + 400 * x[0]**3 + 2*(x[0]-1)
    if n > 2:
        grad[1:n-1] = (-400 * x[1:n-1]*x[2:] + 400 * x[1:n-1]**3 + 2*(x[1:n-1]-1)) \
                      + (200*x[1:n-1] - 200*x[0:n-2]**2)
    grad[n-1] = 200 * x[n-1] - 200 * x[n-2]**2
    return grad

def Rosenbrock_hessian(x):
    n = x.size
    hess_i = np.zeros((n, n))
    hess_ip1 = np.zeros((n, n))
    for i in range(0, n-1):
        hess_i[i, i] = -400 * x[i+1] + 1200 * x[i]**2 + 2
        hess_i[i, i+1] = -400 * x[i]
        hess_ip1[i+1, i] = -400 * x[i]
        hess_ip1[i+1, i+1] = 200 
    return hess_i + hess_ip1

