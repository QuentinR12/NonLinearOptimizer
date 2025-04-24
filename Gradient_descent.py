import numpy as np
import time
from Line_search import Armijo_backtracking, Wolfe_backtracking

def Steepest_descent(f, grad_f, x0, line_search_method, alpha_init, tau, c1, c2, tol, max_iter, max_time, *args):
    """
    Steepest Descent Method.
    Parameters:
    f : callable
        The objective function to minimize.
    grad_f : callable
        The gradient of the objective function.
    x0 : array_like
        Initial guess for the variables.
    alpha_init : float
        Initial step size.
    c1 : float
        Parameter for sufficient decrease condition.
    tau : float
        Reduction factor for step size.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    *args : tuple
        Additional arguments to pass to the objective function and its gradient.
    Returns:
    x_k : array_like
        The point that minimizes the objective function.
    f_val : float
        The value of the objective function at the minimum point.
    k : int
        Number of iterations performed.
    """
    start_time = time.time()
    x_k = np.array(x0, dtype=float)
    f_xk = f(x_k, *args)
    grad_xk = grad_f(x_k, *args)
    norm_grad = np.linalg.norm(grad_xk)
    grad_stop_cond = tol * max(1, norm_grad)
    k = 0

    # tracking history of function values and norm of gradients
    f_history = [f_xk]
    grad_history = [norm_grad]

    # Determine once which line search method to use
    use_wolfe = (line_search_method == 'Wolfe')

    while k < max_iter and norm_grad > grad_stop_cond and time.time() - start_time < max_time:
        # Compute search direction and directional derivative
        p_k = -grad_xk
        dpsi_0 = np.dot(grad_xk, p_k)

        # Choose line search method
        if use_wolfe:
            alpha_k = Wolfe_backtracking(f, grad_f, x_k, f_xk, p_k, dpsi_0, c1, c2, *args)
        else:
            alpha_k = Armijo_backtracking(f, x_k, f_xk, p_k, dpsi_0, alpha_init, c1, tau, *args)

        # Update x
        x_k = x_k + alpha_k * p_k

        # Compute new function value only once
        new_f = f(x_k, *args)
        diff_f = new_f - f_xk
        alpha_init = 2 * diff_f / dpsi_0  # Update step size for next iteration
        
        # Update current values for next iteration
        f_xk = new_f
        grad_xk = grad_f(x_k, *args)
        norm_grad = np.linalg.norm(grad_xk)
        
        f_history.append(f_xk)
        grad_history.append(norm_grad)
        
        k += 1

    elapsed_time = time.time() - start_time
    print(f"Converged in {k} iterations and elapsed time {elapsed_time:.2f} seconds.")
    return x_k, f_xk, k, elapsed_time, f_history, grad_history
