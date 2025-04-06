import numpy as np
import jax.numpy as jnp  # only used in BFGS for norm in your original; you may replace with np.linalg.norm for consistency
from Line_search import Armijo_backtracking, Wolfe_backtracking
from copy import deepcopy
import time

def Cholesky_with_multiple_of_identity(A, beta):
    """
    Returns the Cholesky factor L such that L @ L.T is a positive definite version of A.
    """
    diag_A = np.diag(A)
    delta = 0 if np.min(diag_A) > 0 else np.abs(np.min(diag_A)) + beta
    I = np.eye(A.shape[0])
    
    # Try increasing delta until A + delta*I is positive definite.
    while True:
        try:
            L = np.linalg.cholesky(A + delta * I)
            break
        except np.linalg.LinAlgError:
            delta = max(2 * delta, beta)
    return L

def Newton_modified(f, grad_f, hess_f, x0, line_search_method, alpha_init, tau, c1, c2, beta, tol, max_iter, max_time, *args):
    """
    Modified Newton's Method with Hessian regularization.
    """
    start_time = time.time()
    x_k = np.array(x0, dtype=float)
    f_xk = f(x_k, *args)
    grad_xk = grad_f(x_k, *args)
    norm_grad0 = np.linalg.norm(grad_xk)
    grad_stop_cond = tol * max(1, norm_grad0)
    hess_xk = hess_f(x_k, *args)
    k = 0

    while k < max_iter and np.linalg.norm(grad_xk) > grad_stop_cond and time.time() - start_time < max_time:
        # Regularize Hessian using a Cholesky factorization.
        L_k = Cholesky_with_multiple_of_identity(hess_xk, beta)
        B_k = L_k @ L_k.T
        p_k = np.linalg.solve(B_k, -grad_xk)
        dpsi_0 = np.dot(grad_xk, p_k)

        # Choose line search method.
        if line_search_method == 'Wolfe':
            alpha_k = Wolfe_backtracking(f, grad_f, x_k, f_xk, p_k, dpsi_0, c1, c2, *args)
        else:
            alpha_k = Armijo_backtracking(f, x_k, f_xk, p_k, dpsi_0, alpha_init, c1, tau, *args)

        x_new = x_k + alpha_k * p_k
        f_new = f(x_new, *args)
        diff_f = f_new - f_xk
        alpha_init = 2 * diff_f / dpsi_0  # update step size

        x_k = x_new
        f_xk = f_new
        grad_xk = grad_f(x_k, *args)
        hess_xk = hess_f(x_k, *args)
        k += 1

    elapsed_time = time.time() - start_time
    print(f"Newton_modified converged in {k} iterations and elapsed time {elapsed_time:.2f} seconds.")
    return x_k, f_xk, k, elapsed_time

def BFGS(f, grad_f, x0, line_search_method, alpha_init, tau, c1, c2, eps_min, tol, max_iter, max_time, *args):
    """
    BFGS Quasi-Newton Method.
    """
    start_time = time.time()
    x_k = np.array(x0, dtype=float)
    f_xk = f(x_k, *args)
    grad_xk = grad_f(x_k, *args)
    norm_grad0 = np.linalg.norm(grad_xk)
    grad_stop_cond = tol * max(1, norm_grad0)
    n = len(x_k)
    I = np.eye(n)
    H_k = I.copy()
    k = 0

    while k < max_iter and np.linalg.norm(grad_xk) > grad_stop_cond and time.time() - start_time < max_time:
        p_k = -H_k @ grad_xk
        dpsi_0 = np.dot(grad_xk, p_k)

        if line_search_method == 'Wolfe':
            alpha_k = Wolfe_backtracking(f, grad_f, x_k, f_xk, p_k, dpsi_0, c1, c2, *args)
        else:
            alpha_k = Armijo_backtracking(f, x_k, f_xk, p_k, dpsi_0, alpha_init, c1, tau, *args)

        x_new = x_k + alpha_k * p_k
        s_k = x_new - x_k
        grad_new = grad_f(x_new, *args)
        y_k = grad_new - grad_xk
        yTs = np.dot(y_k, s_k)
        
        if yTs > eps_min * np.linalg.norm(y_k) * np.linalg.norm(s_k):
            rho_k = 1 / yTs
            # BFGS update formula.
            H_k = (I - rho_k * np.outer(s_k, y_k)) @ H_k @ (I - rho_k * np.outer(y_k, s_k)) + rho_k * np.outer(s_k, s_k)
        
        x_k, f_xk, grad_xk = x_new, f(x_new, *args), grad_new
        k += 1

    elapsed_time = time.time() - start_time
    print(f"BFGS converged in {k} iterations and elapsed time {elapsed_time:.2f} seconds.")
    return x_k, f_xk, k, elapsed_time

def L_BFGS(f, grad_f, x0, line_search_method, alpha_init, tau, c1, c2, eps_min, tol, max_iter, max_time, *args):
    """
    L-BFGS Method using limited-memory updates.
    """
    start_time = time.time()
    x_k = np.array(x0, dtype=float)
    f_xk = f(x_k, *args)
    grad_xk = grad_f(x_k, *args)
    norm_grad0 = np.linalg.norm(grad_xk)
    grad_stop_cond = tol * max(1, norm_grad0)
    gamma_k = 1.0
    k = 0
    n = len(x_k)
    m = min(n, 10)
    s_list, y_list, rho_list = [], [], []

    while k < max_iter and np.linalg.norm(grad_xk) > grad_stop_cond and time.time() - start_time < max_time:
        # Two-loop recursion for approximate inverse Hessian product.
        q = grad_xk.copy()
        alpha_vals = []
        L = len(s_list)
        for i in range(L - 1, -1, -1):
            alpha_i = rho_list[i] * np.dot(s_list[i], q)
            q -= alpha_i * y_list[i]
            alpha_vals.append(alpha_i)
        H0 = gamma_k * np.eye(n)
        r = H0 @ q
        for i in range(L):
            beta = rho_list[i] * np.dot(y_list[i], r)
            r += s_list[i] * (alpha_vals[L - 1 - i] - beta)
        p_k = -r
        dpsi_0 = np.dot(grad_xk, p_k)

        if line_search_method == 'Wolfe':
            alpha_k = Wolfe_backtracking(f, grad_f, x_k, f_xk, p_k, dpsi_0, c1, c2, *args)
        else:
            alpha_k = Armijo_backtracking(f, x_k, f_xk, p_k, dpsi_0, alpha_init, c1, tau, *args)

        x_new = x_k + alpha_k * p_k
        s_k = x_new - x_k
        grad_new = grad_f(x_new, *args)
        y_k = grad_new - grad_xk
        gamma_k = np.dot(s_k, y_k) / (np.dot(y_k, y_k) + 1e-12)
        yTs = np.dot(y_k, s_k)
        if yTs > eps_min * np.linalg.norm(y_k) * np.linalg.norm(s_k):
            rho_k = 1 / yTs
            if len(s_list) == m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
            s_list.append(s_k)
            y_list.append(y_k)
            rho_list.append(rho_k)

        x_k, f_xk, grad_xk = x_new, f(x_new, *args), grad_new
        k += 1

    elapsed_time = time.time() - start_time
    print(f"L_BFGS converged in {k} iterations and elapsed time {elapsed_time:.2f} seconds.")
    return x_k, f_xk, k, elapsed_time

def Newton_CG(f, grad_f, hess_f, x0, line_search_method, alpha_init, tau, c1, c2, eta, tol, max_iter, max_time, *args):
    """
    Newton's Method with Conjugate Gradient inner loop.
    """
    start_time = time.time()
    x_k = np.array(x0, dtype=float)
    f_xk = f(x_k, *args)
    grad_xk = grad_f(x_k, *args)
    norm_grad0 = np.linalg.norm(grad_xk)
    hess_xk = hess_f(x_k, *args)
    k = 0

    while k < max_iter and np.linalg.norm(grad_xk) > tol * max(1, norm_grad0) and time.time() - start_time < max_time:
        # Initialize CG variables.
        r = grad_xk.copy()
        d = -r
        z = 0  # will store the cumulative step
        
        j = 0
        while np.dot(d, hess_xk @ d) > 0:
            alpha_j = np.dot(r, r) / np.dot(d, hess_xk @ d)
            z = z + alpha_j * d
            r_new = r + alpha_j * (hess_xk @ d)
            if np.linalg.norm(r_new) < eta * np.linalg.norm(grad_xk):
                p_k = z
                break
            beta_j = np.dot(r_new, r_new) / np.dot(r, r)
            d = -r_new + beta_j * d
            r = r_new
            j += 1
        else:
            # If inner loop did not run or did not break, default to steepest descent.
            p_k = -grad_xk

        dpsi_0 = np.dot(grad_xk, p_k)
        if line_search_method == 'Wolfe':
            alpha_k = Wolfe_backtracking(f, grad_f, x_k, f_xk, p_k, dpsi_0, c1, c2, *args)
        else:
            alpha_k = Armijo_backtracking(f, x_k, f_xk, p_k, dpsi_0, alpha_init, c1, tau, *args)

        x_k = x_k + alpha_k * p_k
        f_xk = f(x_k, *args)
        grad_xk = grad_f(x_k, *args)
        hess_xk = hess_f(x_k, *args)
        k += 1

    elapsed_time = time.time() - start_time
    print(f"Newton_CG converged in {k} iterations and elapsed time {elapsed_time:.2f} seconds.")
    return x_k, f_xk, k, elapsed_time
