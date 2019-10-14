import numpy as np
from mlrose import mlrose

fitness = mlrose.Knapsack(weights=[10, 5, 2, 8, 15], values=[1, 2, 3, 4, 5], max_weight_pct=0.4)
problem = mlrose.DiscreteOpt(length=5, fitness_fn=fitness, maximize=True, max_val=2)

rhc = mlrose.RHCRunner(problem=problem,
                       experiment_name="RHC_final",
                       output_directory="/Users/matthieudivet/Desktop/GaTech/Classes/ML/Assignments/Randomized_optimization/knapsack_problem",
                       seed=None,
                       iteration_list=2 ** np.arange(14),
                       max_attempts=1000,
                       restart_list=[0])
rhc_run_stats, rhc_run_curves = rhc.run()


sa = mlrose.SARunner(problem=problem,
                     experiment_name="SA_final",
                     output_directory="/Users/matthieudivet/Desktop/GaTech/Classes/ML/Assignments/Randomized_optimization/knapsack_problem",
                     seed=None,
                     iteration_list=2 ** np.arange(14),
                     max_attempts=1000,
                     temperature_list=[250],
                     decay_list=[mlrose.ExpDecay])
sa_run_stats, sa_run_curves = sa.run()


ga = mlrose.GARunner(problem=problem,
                     experiment_name="GA_final",
                     output_directory="/Users/matthieudivet/Desktop/GaTech/Classes/ML/Assignments/Randomized_optimization/knapsack_problem",
                     seed=None,
                     iteration_list=2 ** np.arange(14),
                     max_attempts=1000,
                     population_sizes=[200],
                     mutation_rates=[0.3])
ga_run_stats, ga_run_curves = ga.run()


mimic = mlrose.MIMICRunner(problem=problem,
                           experiment_name="MIMIC_final",
                           output_directory="/Users/matthieudivet/Desktop/GaTech/Classes/ML/Assignments/Randomized_optimization/knapsack_problem",
                           seed=None,
                           iteration_list=2 ** np.arange(14),
                           population_sizes=[200],
                           max_attempts=500,
                           keep_percent_list=[0.2],
                           use_fast_mimic=True)
mimic_run_stats, mimic_run_curves = mimic.run()
