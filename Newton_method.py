import numpy as np
import jax.numpy as jnp
from Line_search import Armijo_backtracking, Wolfe_backtracking
from copy import deepcopy
import time

def Newton_method(f, grad_f, hess_f, x0, alpha_init=1, c1=1e-4, tau=0.5, tol=1e-6, max_iter=1000, *args):
    """
    Newton's Method for optimization.
    Parameters:
    f : callable
        The objective function to minimize.
    grad_f : callable
        The gradient of the objective function.
    hess_f : callable
        The Hessian of the objective function.
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
    # print(f"{'Iter':>4}  {'f':>10}  {'||grad||':>10}  {'alpha':>10}")
    x_k = np.array(x0, dtype=float)
    f_xk = f(x_k, *args)
    grad_xk = grad_f(x_k, *args)
    norm_grad_x0 = np.linalg.norm(grad_xk)
    hess_xk = hess_f(x_k, *args)
    diff_x = 1e10
    diff_f = 1e10
    k = 0

    while k < max_iter and np.linalg.norm(grad_xk) > tol * max(1, norm_grad_x0): #   
        p_k = np.linalg.solve(hess_xk, -grad_xk)
        dpsi_0 = grad_xk.T @ p_k

        alpha_k = Armijo_backtracking(f, x_k, f_xk, p_k, dpsi_0, alpha_init, c1, tau, *args)

        x_kp1 = x_k + alpha_k * p_k

        diff_f = (f(x_kp1, *args) - f_xk)
        alpha_init = 2 * diff_f / dpsi_0

        diff_x = x_kp1 - x_k
        x_k = x_kp1

        f_xk = f(x_k, *args)
        grad_xk = grad_f(x_k, *args)
        hess_xk = hess_f(x_k, *args)

        k += 1

        # print(f"{k:4d}  {f_xk:10.2e}  {np.linalg.norm(grad_xk):10.2e}  {alpha_k:10.2e}")
        
    print(f"Converged in {k} iterations.")

    return x_k, f_xk, k

def Newton_modified(f, grad_f, hess_f, x0, line_search_method, alpha_init, tau, c1, c2, beta, tol, max_iter, max_time, *args):
    """
    Modified Newton's Method for optimization.
    Parameters:
    f : callable
        The objective function to minimize.
    grad_f : callable
        The gradient of the objective function.
    hess_f : callable
        The Hessian of the objective function.
    x0 : array_like
        Initial guess for the variables.
    alpha_init : float
        Initial step size.
    c1 : float
        Parameter for sufficient decrease condition.
    tau : float
        Reduction factor for step size.
    beta : float
        Small positive number to add to the diagonal of the Hessian.    
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
    # print(f"{'Iter':>4}  {'f':>10}  {'||grad||':>10}  {'alpha':>10}")
    x_k = np.array(x0, dtype=float)
    f_xk = f(x_k, *args)
    grad_xk = grad_f(x_k, *args)
    norm_grad_x0 = np.linalg.norm(grad_xk)
    grad_stop_cond = tol * max(1, norm_grad_x0)
    hess_xk = hess_f(x_k, *args)
    k = 0

    # Determine once which line search method to use
    use_wolfe = (line_search_method == 'Wolfe')

    while k < max_iter and np.linalg.norm(grad_xk) > grad_stop_cond == tol * max(1, norm_grad_x0) and time.time() - start_time < max_time: 
        L_k = Cholesky_with_multiple_of_identity(hess_xk, beta)
        B_k = L_k @ L_k.T
        p_k = np.linalg.solve(B_k, -grad_xk)
        dpsi_0 = grad_xk.T @ p_k

        if use_wolfe:
            alpha_k = Wolfe_backtracking(f, grad_f, x_k, f_xk, p_k, dpsi_0, c1, c2, *args)
        else:
            alpha_k = Armijo_backtracking(f, x_k, f_xk, p_k, dpsi_0, alpha_init, c1, tau, *args) 

        x_kp1 = x_k + alpha_k * p_k

        diff_f = (f(x_kp1, *args) - f_xk)
        alpha_init = 2 * diff_f / dpsi_0

        diff_x = x_kp1 - x_k
        x_k = x_kp1

        f_xk = f(x_k, *args)
        grad_xk = grad_f(x_k, *args)
        hess_xk = hess_f(x_k, *args)

        k += 1

        # print(f"{k:4d}  {f_xk:10.2e}  {np.linalg.norm(grad_xk):10.2e}  {alpha_k:10.2e}")

    stop_time = time.time()
    elapsed_time  = stop_time - start_time
    print(f"Converged in {k} iterations and and elapsed_time  {elapsed_time:.2f} seconds.")

    return x_k, f_xk, k, elapsed_time

def Cholesky_with_multiple_of_identity(A, beta):
    """
    Cholesky decomposition with a multiple of the identity matrix added.
    Parameters:
    A : array_like
        The matrix to decompose.
    beta : float
        small positive number to add to the diagonal.
    Returns:
    L : array_like
        The lower triangular matrix from the Cholesky decomposition.
    """
    if np.min(np.diag(A)) > 0:
        delta = 0
    else:
        delta = np.abs(np.min(np.diag(A))) + beta

    while True:
        try:
            L = np.linalg.cholesky(A + delta * np.eye(A.shape[0]))
            break

        except np.linalg.LinAlgError:
            delta = np.max([2 * delta, beta])

    return L

def BFGS(f, grad_f, x0, line_search_method, alpha_init, tau, c1, c2, eps_min, tol, max_iter, max_time, *args):
    """
    BFGS Method for optimization.
    Parameters:
    f : callable
        The objective function to minimize.
    grad_f : callable
        The gradient of the objective function.
    x0 : array_like
        Initial guess for the variables.
    c1 : float
        Parameter for sufficient decrease condition.
    c2 : float
        Parameter for curvature condition.
    eps_min : float
        Small positive number to ensure positive definiteness.
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
    # print(f"{'Iter':>4}  {'f':>10}  {'||grad||':>10}  {'alpha':>10}")
    x_k = x0
    f_xk = f(x_k, *args)
    grad_xk = grad_f(x_k, *args)
    norm_grad_x0 = jnp.linalg.norm(grad_xk)
    grad_stop_cond = tol * max(1, norm_grad_x0)
    n = len(x_k)
    I = np.eye(n)
    H_k = I
    k = 0

    # Determine once which line search method to use
    use_wolfe = (line_search_method == 'Wolfe')

    while k < max_iter and np.linalg.norm(grad_xk) > grad_stop_cond and time.time() - start_time < max_time:
        p_k = -H_k @ grad_xk

        dpsi_0 = grad_xk.T @ p_k
        if use_wolfe:
            alpha_k = Wolfe_backtracking(f, grad_f, x_k, f_xk, p_k, dpsi_0, c1, c2, *args)
        else:
            alpha_k = Armijo_backtracking(f, x_k, f_xk, p_k, dpsi_0, alpha_init, c1, tau, *args) 

        x_kp1 = x_k + alpha_k * p_k

        s_k = x_kp1 - x_k
        grad_xkp1 = grad_f(x_kp1, *args)
        y_k = grad_xkp1 - grad_xk

        yTs = y_k.T @ s_k
        if yTs > eps_min * np.linalg.norm(y_k) * np.linalg.norm(s_k):
            rho_k = 1 / (yTs)   
            H_k = (I - rho_k * np.outer(s_k, y_k.T)) @ H_k @ (I - rho_k * np.outer(y_k, s_k.T)) + rho_k * np.outer(s_k, s_k.T)

        x_k = x_kp1
        f_xk = f(x_k, *args)
        grad_xk = deepcopy(grad_xkp1)
        k += 1
        # print(f"{k:4d}  {f_xk:10.2e}  {np.linalg.norm(grad_xk):10.2e}  {alpha_k:10.2e}")

    stop_time = time.time()
    elapsed_time  = stop_time - start_time
    print(f"Converged in {k} iterations and and elapsed_time  {elapsed_time:.2f} seconds.")

    return x_k, f_xk, k, elapsed_time

def L_BFGS(f, grad_f, x0, line_search_method, alpha_init, tau, c1, c2, eps_min, tol, max_iter, max_time, *args):
    """
    L-BFGS Method for optimization.
    Parameters:
    f : callable
        The objective function to minimize.
    grad_f : callable
        The gradient of the objective function.
    x0 : array_like
        Initial guess for the variables.
    c1 : float
        Parameter for sufficient decrease condition.
    c2 : float
        Parameter for curvature condition.
    eps_min : float
        Small positive number to ensure positive definiteness.
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
    # print(f"{'Iter':>4}  {'f':>10}  {'||grad||':>10}  {'alpha':>10}")
    x_k = np.array(x0, dtype=float)
    f_xk = f(x_k, *args)
    grad_xk = grad_f(x_k, *args)
    norm_grad_x0 = np.linalg.norm(grad_xk)
    grad_stop_cond = tol * max(1, norm_grad_x0)
    gamma_k = 1 
    k = 0

    n = len(x_k)
    m = min(n, 10)
    s_list = []
    y_list = []
    rho_list = []

    # Determine once which line search method to use
    use_wolfe = (line_search_method == 'Wolfe')

    while k < max_iter and np.linalg.norm(grad_xk) > grad_stop_cond and time.time() - start_time < max_time: 
        q_k = grad_xk.copy()
        alpha_list = []

        L = len(s_list)
        for i in range(L-1, -1, -1):
            alpha_i = rho_list[i] * (s_list[i].T @ q_k)
            q_k -= alpha_i * y_list[i]
            alpha_list.append(alpha_i)

        H0_k = gamma_k * np.eye(n)
        r = H0_k @ q_k
        
        alpha_rev = alpha_list[::-1]
        for i in range(L):
            beta = rho_list[i] * (y_list[i].T @ r)
            r += s_list[i] * (alpha_rev[i] - beta)

        p_k = -r
        dpsi_0 = grad_xk.T @ p_k

        if use_wolfe:
            alpha_k = Wolfe_backtracking(f, grad_f, x_k, f_xk, p_k, dpsi_0, c1, c2, *args)
        else:
            alpha_k = Armijo_backtracking(f, x_k, f_xk, p_k, dpsi_0, alpha_init, c1, tau, *args) 


        x_kp1 = x_k + alpha_k * p_k

        s_k = x_kp1 - x_k
        grad_xkp1 = grad_f(x_kp1, *args)
        y_k = grad_xkp1 - grad_xk
        gamma_k = s_k.T @ y_k / (y_k.T @ y_k)

        yTs = y_k.T @ s_k
        if yTs > eps_min * np.linalg.norm(y_k) * np.linalg.norm(s_k):
            rho_k = 1 / (yTs)
            if k > m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)

            s_list.append(s_k)
            y_list.append(y_k)
            rho_list.append(rho_k)

        x_k = x_kp1
        f_xk = f(x_k, *args)
        grad_xk = grad_xkp1
        k += 1
        # print(f"{k:4d}  {f_xk:10.2e}  {np.linalg.norm(grad_xk):10.2e}  {alpha_k:10.2e}")

    stop_time = time.time()
    elapsed_time  = stop_time - start_time
    print(f"Converged in {k} iterations and and elapsed_time  {elapsed_time:.2f} seconds.")

    return x_k, f_xk, k, elapsed_time

def Newton_CG(f, grad_f, hess_f, x0, line_search_method, alpha_init, tau, c1, c2, eta, tol, max_iter, max_time, *args):
    """
    Newton's Method with Conjugate Gradient for optimization.
    Parameters:
    f : callable
        The objective function to minimize.
    grad_f : callable
        The gradient of the objective function.
    hess_f : callable
        The Hessian of the objective function.
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
    # print(f"{'Iter':>4}  {'f':>10}  {'||grad||':>10}  {'alpha':>10}")
    x_k = np.array(x0, dtype=float)
    f_xk = f(x_k, *args)
    grad_xk = grad_f(x_k, *args)
    norm_grad_x0 = np.linalg.norm(grad_xk)
    hess_xk = hess_f(x_k, *args)
    diff_x = 1e10
    diff_f = 1e10
    k = 0

    # Determine once which line search method to use
    use_wolfe = (line_search_method == 'Wolfe')

    while k < max_iter and np.linalg.norm(grad_xk) > tol * max(1, norm_grad_x0) and time.time() - start_time < max_time:
        j = 0
        z_j = 0
        r_j = grad_xk
        d_j = -r_j
        while d_j.T @ hess_xk @ d_j > 0:
            alpha_j = (r_j.T @ r_j) / (d_j.T @ hess_xk @ d_j)
            z_j = z_j + alpha_j * d_j
            r_jp1 = r_j + alpha_j * hess_xk @ d_j

            if np.linalg.norm(r_jp1) < eta * np.linalg.norm(grad_xk):
                p_k = z_j
                break 

            beta_j = (r_jp1.T @ r_jp1) / (r_j.T @ r_j)
            d_j = -r_jp1 + beta_j * d_j
            r_j = r_jp1
            j += 1

        if j == 0:
            p_k = -grad_xk
        else:
            p_k = z_j
            
        dpsi_0 = grad_xk.T @ p_k
        if use_wolfe:
            alpha_k = Wolfe_backtracking(f, grad_f, x_k, f_xk, p_k, dpsi_0, c1, c2, *args)
        else:
            alpha_k = Armijo_backtracking(f, x_k, f_xk, p_k, dpsi_0, alpha_init, c1, tau, *args) 

        x_kp1 = x_k + alpha_k * p_k

        x_k = x_kp1

        f_xk = f(x_k, *args)
        grad_xk = grad_f(x_k, *args)
        hess_xk = hess_f(x_k, *args)

        k += 1

        # print(f"{k:4d}  {f_xk:10.2e}  {np.linalg.norm(grad_xk):10.2e}  {alpha_k:10.2e}")

    stop_time = time.time()
    elapsed_time  = stop_time - start_time
    print(f"Converged in {k} iterations and and elapsed_time  {elapsed_time:.2f} seconds.")

    return x_k, f_xk, k, elapsed_time
