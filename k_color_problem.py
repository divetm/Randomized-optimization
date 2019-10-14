import numpy as np
from mlrose import mlrose

edges = [(0, 1), (1, 2), (0, 2), (1, 3), (2, 3), (3, 4)]
fitness = mlrose.MaxKColor(edges)
problem = mlrose.DiscreteOpt(length=5, fitness_fn=fitness, maximize=False, max_val=2)

rhc = mlrose.RHCRunner(problem=problem,
                       experiment_name="RHC_final",
                       output_directory="/Users/matthieudivet/Desktop/GaTech/Classes/ML/Assignments/Randomized_optimization/k_color_problem",
                       seed=None,
                       iteration_list=2 ** np.arange(14),
                       max_attempts=1000,
                       restart_list=[0])
rhc_run_stats, rhc_run_curves = rhc.run()


sa = mlrose.SARunner(problem=problem,
                     experiment_name="SA_final",
                     output_directory="/Users/matthieudivet/Desktop/GaTech/Classes/ML/Assignments/Randomized_optimization/k_color_problem",
                     seed=None,
                     iteration_list=2 ** np.arange(14),
                     max_attempts=1000,
                     temperature_list=[250],
                     decay_list=[mlrose.ExpDecay])
sa_run_stats, sa_run_curves = sa.run()


ga = mlrose.GARunner(problem=problem,
                     experiment_name="GA_final",
                     output_directory="/Users/matthieudivet/Desktop/GaTech/Classes/ML/Assignments/Randomized_optimization/k_color_problem",
                     seed=None,
                     iteration_list=2 ** np.arange(14),
                     max_attempts=1000,
                     population_sizes=[200],
                     mutation_rates=[0.3])
ga_run_stats, ga_run_curves = ga.run()


mimic = mlrose.MIMICRunner(problem=problem,
                           experiment_name="MIMIC_final",
                           output_directory="/Users/matthieudivet/Desktop/GaTech/Classes/ML/Assignments/Randomized_optimization/k_color_problem",
                           seed=None,
                           iteration_list=2 ** np.arange(14),
                           population_sizes=[200],
                           max_attempts=500,
                           keep_percent_list=[0.2],
                           use_fast_mimic=True)
mimic_run_stats, mimic_run_curves = mimic.run()
