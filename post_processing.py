
import numpy as np
from src.plotting import plot_stuff

import numpy as np
from src.plotting import plot_stuff

# Custom's selection
exp = 'exp_synthetic_3_clusters_SMALL'  # Figure 1 left
# exp = 'exp_synthetic_3_clusters_LARGE'  # Figure 1 right
# exp = 'exp_real_lenk'  # Figure 2 left
# exp = 'exp_real_movies'  # Figure 2 middle
# exp = 'exp_real_jester'  # Figure 2 right

if exp == 'exp_synthetic_3_clusters_SMALL':
    methods = ['ITL']
    saved_file_name = 'saved_results/synth_3_clusters_SMALL_results.npy'
    dataset = 'synthetic-regression-3-CLUSTERS-SMALL'
elif exp == 'exp_synthetic_3_clusters_LARGE':
    methods = ['ITL']
    saved_file_name = 'saved_results/synth_3_clusters_LARGE_results.npy'
    dataset = 'synthetic-regression-3-CLUSTERS-LARGE'
elif exp == 'exp_real_lenk':
    methods = ['ITL']
    saved_file_name = 'saved_results/lenk_results.npy'
    dataset = 'lenk'
elif exp == 'exp_real_movies':
    methods = ['ITL']
    saved_file_name = 'saved_results/movies_results_2.npy'
    dataset = 'movies'
elif exp == 'exp_real_jester':
    methods = ['ITL']
    saved_file_name = 'saved_results/jester_results_2.npy'
    dataset = 'jester'

results = np.load(saved_file_name, allow_pickle='TRUE').item()
plot_stuff(results, methods, dataset)