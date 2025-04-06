import numpy as np
from copy import deepcopy

def Armijo_backtracking(f, x_k, f_xk, p_k, dpsi_0, alpha_init, c1, tau, *args):
    """
    Armijo backtracking line search for step size selection.
    Parameters:
    f : callable
        The objective function to minimize.
    x_k : array_like
        Current point.
    f_xk : float
        Function value at x_k.
    p_k : array_like
        Search direction.
    dpsi_0 : float
        Directional derivative at x_k.
    alpha_init : float
        Initial step size.
    c1 : float
        Parameter for sufficient decrease condition.
    tau : float
        Reduction factor for step size.
    Returns:
    alpha : float
        Step size that satisfies the Armijo condition.
    """
    alpha = alpha_init

    while f(x_k + alpha * p_k, *args) > f_xk + c1 * alpha * dpsi_0:
        alpha *= tau

    return alpha

def Wolfe_backtracking(f, grad_f, x_k, f_xk, p_k, dpsi_0, c1, c2, *args):
    """
    Wolfe backtracking line search for step size selection.
    Parameters:
    f : callable
        The objective function to minimize.
    grad_f : callable
        The gradient of the objective function.
    x_k : array_like
        Current point.
    f_xk : float
        Function value at x_k.
    p_k : array_like
        Search direction.
    dpsi_0 : float
        Directional derivative at x_k.
    c1 : float
        Parameter for sufficient decrease condition.
    c2 : float
        Parameter for curvature condition.
    Returns:
    alpha : float
        Step size that satisfies the Wolfe conditions.
    """
    alpha = 1
    # prev_alpha = 0
    alpha_l = 0
    alpha_u = np.inf

    x_kp1 = x_k + alpha * p_k
    armijo_cond = f(x_kp1, *args) <= f_xk + c1 * alpha * dpsi_0
    curvature_cond = grad_f(x_kp1, *args) @ p_k >= c2 * dpsi_0

    # steps = 0
    # counter = 0
    while (not armijo_cond or not curvature_cond):
        
        if not armijo_cond:
            alpha_u = deepcopy(alpha)
        else:
            if not curvature_cond:
                alpha_l = deepcopy(alpha)

        if alpha_u < np.inf:
            alpha = (alpha_l + alpha_u) / 2
        else:
            alpha = 2 * alpha

        # if alpha <= 1e-6:
        #     break

        # if abs(alpha - prev_alpha) <= 1e-8: 
        #     counter += 1
        #     if counter > 20:
        #         break
        #     break

        x_kp1 = x_k + alpha * p_k
        armijo_cond = f(x_kp1, *args) <= f_xk + c1 * alpha * dpsi_0
        curvature_cond = grad_f(x_kp1, *args) @ p_k >= c2 * dpsi_0

        # prev_alpha = deepcopy(alpha)
        # steps += 1
    return alpha