import numpy as np
import pandas as pd
from Rosenbrock_function import Rosenbrock, Rosenbrock_gradient, Rosenbrock_hessian
from Beale_function import beale, beale_gradient, beale_hessian
from The_function import the_function, the_function_gradient, the_function_hessian
from Line_search import Armijo_backtracking, Wolfe_backtracking
from Gradient_descent import Steepest_descent
from Newton_method import Newton_modified, Newton_CG, BFGS, L_BFGS

# Classes as defined before
class Problem:
    def __init__(self, x0, name):
        self.x0 = x0
        self.name = name

class Method:
    def __init__(self, name, beta=1e-4, eps_min=1e-8, eta=1e-2, tol=1e-9, max_iter=5000, max_time=1e6, *args):
        self.name = name
        self.beta = beta
        self.eps_min = eps_min
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.max_time = max_time
        self.args = args

class Line_search:
    def __init__(self, name='Armijo', alpha=1, tau=0.5, c1=1e-4, c2=0.9, *args):
        self.name = name
        self.alpha = alpha
        self.tau = tau
        self.c1 = c1
        self.c2 = c2
        self.args = args

class OptSolver_KFC:
    def __init__(self, problem, method, line_search, *args):
        if not isinstance(problem, Problem):
            raise ValueError("Invalid problem instance")
        self.x0 = problem.x0
        self.problem = problem
        self.method = method
        self.line_search = line_search
        
        # Set function, gradient, and Hessian based on problem type
        if self.problem.name == 'Rosenbrock':
            self.f = Rosenbrock
            self.grad = Rosenbrock_gradient
            self.hess = Rosenbrock_hessian
        elif self.problem.name == 'Beale':
            self.f = beale
            self.grad = beale_gradient
            self.hess = beale_hessian
        elif self.problem.name == 'the_function':
            self.f = the_function
            self.grad = the_function_gradient
            self.hess = the_function_hessian
        else:
            raise ValueError("Unknown problem name")
        
        # Select the optimization method based on method name
        if self.method.name == 'GradientDescent':
            self.x, self.f_x, self.n_iter, self.elapsed_time = Steepest_descent(
                self.f, self.grad, self.x0, self.line_search.name, self.line_search.alpha,
                self.line_search.tau, self.line_search.c1, self.line_search.c2,
                self.method.tol, self.method.max_iter, self.method.max_time, *args)
        elif self.method.name == 'ModifiedNewton':
            self.x, self.f_x, self.n_iter, self.elapsed_time = Newton_modified(
                self.f, self.grad, self.hess, self.x0, self.line_search.name, self.line_search.alpha,
                self.line_search.tau, self.line_search.c1, self.line_search.c2,
                self.method.beta, self.method.tol, self.method.max_iter, self.method.max_time, *args)
        elif self.method.name == 'NewtonCG':
            self.x, self.f_x, self.n_iter, self.elapsed_time = Newton_CG(
                self.f, self.grad, self.hess, self.x0, self.line_search.name, self.line_search.alpha,
                self.line_search.tau, self.line_search.c1, self.line_search.c2,
                self.method.eta, self.method.tol, self.method.max_iter, self.method.max_time, *args)
        elif self.method.name == 'BFGS':
            self.x, self.f_x, self.n_iter, self.elapsed_time = BFGS(
                self.f, self.grad, self.x0, self.line_search.name, self.line_search.alpha,
                self.line_search.tau, self.line_search.c1, self.line_search.c2,
                self.method.eps_min, self.method.tol, self.method.max_iter, self.method.max_time, *args)
        elif self.method.name == 'L_BFGS':
            self.x, self.f_x, self.n_iter, self.elapsed_time = L_BFGS(
                self.f, self.grad, self.x0, self.line_search.name, self.line_search.alpha,
                self.line_search.tau, self.line_search.c1, self.line_search.c2,
                self.method.eps_min, self.method.tol, self.method.max_iter, self.method.max_time, *args)
        else:
            raise ValueError("Unknown method name")