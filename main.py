import numpy as np
import pandas as pd
from Solver_KFC import Problem, Method, Line_search, OptSolver_KFC

# Define the solving methods
method_GradientDescent = Method(name='GradientDescent', max_iter=5000, tol=1e-9, max_time=300)
method_ModifiedNewton = Method(name='ModifiedNewton', beta=1e-4, max_iter=5000, tol=1e-9, max_time=300)
method_NewtonCG = Method(name='NewtonCG', eta=1e-2, max_iter=5000, tol=1e-9, max_time=300)
method_BFGS = Method(name='BFGS', eps_min=1e-8, max_iter=5000, tol=1e-9, max_time=300)
method_L_BFGS = Method(name='L_BFGS', eps_min=1e-8, max_iter=5000, tol=1e-9, max_time=300)

# Define the line search methods
line_search_Armijo = Line_search(name='Armijo', alpha=1, c1=1e-4, tau=0.5)
line_search_Wolfe = Line_search(name='Wolfe', alpha=1, c1=1e-4, c2=0.9)

# Define the problems
problems = [
    ('Problem 1', Problem(x0=np.array([-1.2, 1]), name='Rosenbrock')),
    ('Problem 2', Problem(x0=np.ones(10) * -1, name='Rosenbrock')),
    ('Problem 3', Problem(x0=np.ones(10) * 2, name='Rosenbrock')),
    ('Problem 4', Problem(x0=np.ones(100) * -1, name='Rosenbrock')),
    ('Problem 5', Problem(x0=np.ones(100) * 2, name='Rosenbrock')),
    ('Problem 6', Problem(x0=np.ones(1000) * 2, name='Rosenbrock')),
    ('Problem 7', Problem(x0=np.ones(10000) * 2, name='Rosenbrock')),
    ('Problem 8', Problem(x0=np.array([1, 1]), name='Beale')),
    ('Problem 9', Problem(x0=np.array([0, 0]), name='Beale')),
    ('Problem 10', Problem(x0=np.arange(1, 11), name='the_function')),
]

methods = [
    ('GradientDescent', method_GradientDescent),
    ('ModifiedNewton', method_ModifiedNewton),
    ('NewtonCG', method_NewtonCG),
    ('BFGS', method_BFGS),
    ('L_BFGS', method_L_BFGS),
]

line_searches = [
    ('Armijo', line_search_Armijo),
    ('Wolfe', line_search_Wolfe),
]

# List to store the results
results_list = []

# Iterate over every combination of problem, method, and line search
for q_name, prob in problems:
    for m_name, method in methods:
        for ls_name, ls in line_searches:
            # Print a fancy header for clarity
            print("\n" + "=" * 80)
            print(f"Solving: {q_name}")
            print(f"Method: {method.name}")
            print(f"Line Search: {ls_name}")
            print("=" * 80 + "\n")
            
            try:
                optimizer = OptSolver_KFC(prob, method, ls)
                # Simple termination indicator (can be replaced with more sophisticated logic)
                termination = "Converged" if optimizer.n_iter < method.max_iter else "Max iterations reached"
                # Placeholder for Newton's method modification relevance
                newton_relevance = "Relevant" if method.name == 'ModifiedNewton' else "N/A"
                # Placeholder for local rate of convergence (if available)
                local_rate = "N/A"
                
                results_list.append({
                    'Question': q_name,
                    'Method': f"{method.name} + {ls_name}",
                    'x_min': optimizer.x,
                    'f_min': optimizer.f_x,
                    'Number of iterations': optimizer.n_iter,
                    'Elapsed Time': optimizer.elapsed_time,
                    'Termination': termination,
                    "Local rate of convergence": local_rate
                })
            except Exception as e:
                # In case of an error, store the error message in the Termination column
                results_list.append({
                    'Question': q_name,
                    'Method': f"{method.name} + {ls_name}",
                    'x_min': None,
                    'f_min': None,
                    'Number of iterations': None,
                    'Elapsed Time': None,
                    'Termination': f"Error: {e}",
                    "Local rate of convergence": "N/A"
                })

# Create the DataFrame with the desired columns
results = pd.DataFrame(results_list, columns=[
    'Question', 'Method', 'x_min', 'f_min', 'Number of iterations',
    'Elapsed Time', 'Termination', "Local rate of convergence"
])

# Save the resulting DataFrame
results.to_csv('results.csv', index=False)