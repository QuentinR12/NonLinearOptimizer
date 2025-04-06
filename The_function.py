import numpy as np

def the_function(x, n):
    return x[0]**2 + sum((x[i]-x[i+1])**(2*(i+1)) for i in range(0, n-1))

def the_function_gradient(x, n):
    grad = np.zeros(n)
    grad[0] = 2*x[0]
    for i in range(0, n-1):
        grad[i] += 2*(i+1)*(x[i]-x[i+1])**(2*(i+1)-1)
        grad[i+1] -= 2*(i+1)*(x[i]-x[i+1])**(2*(i+1)-1)
    return grad

def the_function_hessian(x, n):
    H = np.zeros((n, n))
    H[0, 0] = 2
    for i in range(0, n-1):
        factor = 2*(i+1)*(2*(i+1)-1)*(x[i]-x[i+1])**(2*(i+1)-2)
        H[i, i] += factor
        H[i, i+1] -= factor
        H[i+1, i] -= factor
        H[i+1, i+1] += factor
    return H