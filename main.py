import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Solver_KFC import Problem, Method, Line_search, OptSolver_KFC

# ensure output folder exists
os.makedirs('figures', exist_ok=True)

# Define the solving methods
method_GradientDescent = Method(name='GradientDescent', max_iter=5000, tol=1e-9, max_time=300)
method_ModifiedNewton = Method(name='ModifiedNewton', beta=1e-4, max_iter=5000, tol=1e-9, max_time=300)
method_NewtonCG       = Method(name='NewtonCG',       eta=1e-2,  max_iter=5000, tol=1e-9, max_time=300)
method_BFGS           = Method(name='BFGS',           eps_min=1e-8, max_iter=5000, tol=1e-9, max_time=300)
method_L_BFGS         = Method(name='L_BFGS',         eps_min=1e-8, max_iter=5000, tol=1e-9, max_time=300)
method_DFP            = Method(name='DFP',            eps_min=1e-8, max_iter=5000, tol=1e-9, max_time=300)

# Define the line search methods
line_search_Armijo = Line_search(name='Armijo', alpha=1, c1=1e-4, tau=0.5)
line_search_Wolfe  = Line_search(name='Wolfe',  alpha=1, c1=1e-4, c2=0.9)

# Define the problems
problems = [
    ('Problem 1',  Problem(x0=np.array([-1.2, 1]),               name='Rosenbrock')),
    ('Problem 2',  Problem(x0=np.ones(10) * -1,                 name='Rosenbrock')),
    ('Problem 3',  Problem(x0=np.ones(10) * 2,                  name='Rosenbrock')),
    ('Problem 4',  Problem(x0=np.ones(100) * -1,                name='Rosenbrock')),
    ('Problem 5',  Problem(x0=np.ones(100) * 2,                 name='Rosenbrock')),
    ('Problem 6',  Problem(x0=np.ones(1000) * 2,                name='Rosenbrock')),
    ('Problem 7',  Problem(x0=np.ones(10000) * 2,               name='Rosenbrock')),
    ('Problem 8',  Problem(x0=np.array([1, 1]),                 name='Beale')),
    ('Problem 9',  Problem(x0=np.array([0, 0]),                 name='Beale')),
    ('Problem 10', Problem(x0=np.arange(1, 11),                 name='the_function')),
]

methods = [
    ('GradientDescent', method_GradientDescent),
    ('ModifiedNewton',  method_ModifiedNewton),
    ('NewtonCG',        method_NewtonCG),
    ('BFGS',            method_BFGS),
    ('L_BFGS',          method_L_BFGS),
    ('DFP',             method_DFP),
]

line_searches = [
    ('Armijo', line_search_Armijo),
    ('Wolfe',  line_search_Wolfe),
]

results_list = []

for q_name, prob in problems:
    # sanitize problem name for filenames
    fname_base = q_name.replace(' ', '_').lower()

    # Create two figures for this problem
    fig_f, ax_f       = plt.subplots()
    fig_grad, ax_grad = plt.subplots()

    for m_name, method in methods:
        for ls_name, ls in line_searches:
            print(f"\n=== {q_name} | {method.name} + {ls_name} ===\n")
            # try:
            optimizer = OptSolver_KFC(prob, method, ls)

            # record results
            termination = ("Converged" if optimizer.n_iter < method.max_iter 
                            else "Max iterations")
            results_list.append({
                'Question': q_name,
                'Method':   f"{method.name} + {ls_name}",
                'x_min':    optimizer.x,
                'f_min':    optimizer.f_x,
                'Number of iterations': optimizer.n_iter,
                'Elapsed Time':         optimizer.elapsed_time,
                'Termination':          termination,
                "Local rate of convergence": "N/A"
            })

            # --- PLOTTING ---
            iters = np.arange(0, optimizer.n_iter + 1)

            # (i) function value vs iteration
            ax_f.plot(iters, optimizer.f_history,
                        label=f"{method.name}+{ls_name}")

            # (ii) norm of gradient (log-log)
            grad_norms = [np.linalg.norm(g) for g in optimizer.grad_history]
            ax_grad.loglog(iters, grad_norms,
                            label=f"{method.name}+{ls_name}")

            # except Exception as e:
                # results_list.append({
                #     'Question': q_name,
                #     'Method':   f"{method.name} + {ls_name}",
                #     'x_min':    None,
                #     'f_min':    None,
                #     'Number of iterations': None,
                #     'Elapsed Time':         None,
                #     'Termination':          f"Error: {e}",
                #     "Local rate of convergence": "N/A"
                # })

    # finalize & save the function‐value plot
    ax_f.set_title   (f"{q_name}: f(x) vs Iteration")
    ax_f.set_xlabel  ("Iteration")
    ax_f.set_ylabel  ("f(x)")
    ax_f.grid        (True)
    # only draw a legend if there's something to show
    handles, labels = ax_f.get_legend_handles_labels()
    if handles:
        ax_f.legend(loc='best')
    fig_f.tight_layout()
    fig_f.savefig(f"figures/{fname_base}_f_vs_iter.png", dpi=300)
    # plt.show()

    # finalize & save the gradient‐norm plot
    ax_grad.set_title  (f"{q_name}: ‖∇f‖ vs Iteration (log–log)")
    ax_grad.set_xlabel ("Iteration")
    ax_grad.set_ylabel ("‖∇f‖")
    ax_grad.grid       (True, which='both')
    handles, labels = ax_grad.get_legend_handles_labels()
    if handles:
        ax_grad.legend(loc='best')
    fig_grad.tight_layout()
    fig_grad.savefig(f"figures/{fname_base}_grad_vs_iter.png", dpi=300)
    # plt.show()

# build results DataFrame
results = pd.DataFrame(results_list, columns=[
    'Question', 'Method', 'x_min', 'f_min',
    'Number of iterations', 'Elapsed Time',
    'Termination', "Local rate of convergence"
])
results.to_csv('results.csv', index=False)
