import numpy as np
from mlrose import mlrose

edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (1, 5), (2, 5)]
fitness = mlrose.MaxKColor(edges)
problem = mlrose.DiscreteOpt(length=6, fitness_fn=fitness, maximize=False, max_val=3)

rhc = mlrose.RHCRunner(problem=problem,
                       experiment_name="different_restarts",
                       output_directory="/Users/matthieudivet/Desktop/GaTech/Classes/ML/Assignments/Randomized_optimization/k_color_problem/RHC",
                       seed=None,
                       iteration_list=2 ** np.arange(12),
                       max_attempts=1000,
                       restart_list=[0, 10, 100])
rhc_run_stats, rhc_run_curves = rhc.run()


sa = mlrose.SARunner(problem=problem,
                     experiment_name="different_decays",
                     output_directory="/Users/matthieudivet/Desktop/GaTech/Classes/ML/Assignments/Randomized_optimization/k_color_problem/SA",
                     seed=None,
                     iteration_list=2 ** np.arange(12),
                     max_attempts=1000,
                     temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
                     decay_list=[mlrose.ExpDecay, mlrose.GeomDecay, mlrose.ArithDecay])
sa_run_stats, sa_run_curves = sa.run()


ga = mlrose.GARunner(problem=problem,
                     experiment_name="different_pop_sizes_and_mutation_pct",
                     output_directory="/Users/matthieudivet/Desktop/GaTech/Classes/ML/Assignments/Randomized_optimization/k_color_problem/GA",
                     seed=None,
                     iteration_list=2 ** np.arange(12),
                     max_attempts=1000,
                     population_sizes=[150, 200, 300],
                     mutation_rates=[0.1, 0.3, 0.5, 0.6])
ga_run_stats, ga_run_curves = ga.run()


mimic = mlrose.MIMICRunner(problem=problem,
                           experiment_name="different_pop_sizes_and_kp_pct",
                           output_directory="/Users/matthieudivet/Desktop/GaTech/Classes/ML/Assignments/Randomized_optimization/k_color_problem/MIMIC",
                           seed=None,
                           iteration_list=2 ** np.arange(12),
                           population_sizes=[200, 300, 500],
                           max_attempts=500,
                           keep_percent_list=[0.2, 0.3, 0.5],
                           use_fast_mimic=True)
mimic_run_stats, mimic_run_curves = mimic.run()
