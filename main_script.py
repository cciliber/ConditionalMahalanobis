

import numpy as np
import matplotlib
from src.data_management import DataHandler, Settings
from src.methods import FixedMahalanobis, MetaLearningMahalanobis
from src.plotting import plot_stuff
import time
import datetime
from itertools import product


def main():

    # Custom's selection
    # exp = 'exp_synthetic_3_clusters_SMALL'  # Figure 1 left
    exp = 'exp_synthetic_3_clusters_LARGE'  # Figure 1 right

    # exp = 'exp_real_lenk'  # Figure 2 left
    # exp = 'exp_real_movies'  # Figure 2 middle
    # exp = 'exp_real_jester'  # Figure 2 right

    if exp == 'exp_synthetic_3_clusters_SMALL':
        data_settings = {'dataset': 'synthetic-regression-3-CLUSTERS-SMALL',
                         'n_tr_tasks': 200,  # 500,
                         'n_val_tasks': 100,  # 300,
                         'n_test_tasks': 100,  # 100,
                         'n_all_points': 50,  # 80,
                         'ts_points_pct': 0.5,
                         'n_dims': 18,   # 20,
                         'noise_std': 0.1,
                         'number_clusters': 3,
                         'sparsity': 6}
        loss_name = 'absolute'
        feature_map_name_bias = 'linear_with_labels'
        feature_map_name_feature = 'linear_with_labels'
        r = None
        W = None
        # methods = ['ITL', 'unconditional_Bias', 'conditional_Bias', 'unconditional_Feature', 'conditional_Feature', 'unconditional_Mahalanobis', 'conditional_Mahalanobis']
        methods = ['ITL', 'unconditional_Bias', 'conditional_Bias']
        results = {}
    if exp == 'exp_synthetic_3_clusters_LARGE':
        data_settings = {'dataset': 'synthetic-regression-3-CLUSTERS-LARGE',
                         'n_tr_tasks': 500,
                         'n_val_tasks': 300,
                         'n_test_tasks': 100,
                         'n_all_points': 80,
                         'ts_points_pct': 0.5,
                         'n_dims': 18,
                         'noise_std': 0.1,
                         'number_clusters': 3,
                         'sparsity': 6}
        loss_name = 'absolute'
        feature_map_name_bias = 'linear_with_labels'
        feature_map_name_feature = 'linear_with_labels'
        r = None
        W = None
        methods = ['ITL', 'unconditional_Bias', 'conditional_Bias', 'unconditional_Feature', 'conditional_Feature', 'unconditional_Mahalanobis', 'conditional_Mahalanobis']
        results = {}
    elif exp == 'exp_real_lenk':
        data_settings = {'dataset': 'lenk',
                         'n_tr_tasks': 100,
                         'n_val_tasks': 40,
                         'n_test_tasks': 30,
                         }
        loss_name = 'absolute'
        feature_map_name_bias = 'ls_regressor'
        feature_map_name_feature = 'ls_regressor'
        r = None
        W = None
        methods = ['ITL']
        results = {}
    elif exp == 'exp_real_movies':
        data_settings = {'dataset': 'movies',  # 943 tasks in total, n_tot = d = 939
                         'n_tr_tasks': 200,  # 400,  # 700,
                         'n_val_tasks': 100,  # 100,  # 100,
                         'n_test_tasks': 100,  # 100,  # 143,
                         'ts_points_pct': 0.25
                         }
        loss_name = 'absolute'
        feature_map_name_bias = 'linear_with_labels'
        feature_map_name_feature = 'linear_with_labels'
        r = 5
        W = None
        methods = ['ITL']
        results = {}
    elif exp == 'exp_real_jester':
        data_settings = {'dataset': 'jester',
                         'n_tr_tasks': 250,
                         'n_val_tasks': 100,
                         'n_test_tasks': 100,
                         'ts_points_pct': 0.25
                         }
        loss_name = 'absolute'
        feature_map_name_bias = 'recommenders'
        feature_map_name_feature = 'recommenders'
        r = 5  # in the case of recommender systems, we put the score in r
        W = None
        methods = ['ITL']
        results = {}

    font = {'size': 26}
    matplotlib.rc('font', **font)

    for curr_method in methods:

        results[curr_method] = []

    tt = time.time()

    trials = 1  # 5

    for seed in range(trials):

        np.random.seed(seed)
        general_settings = {'seed': seed, 'verbose': 1}
        settings = Settings(data_settings, 'data')
        settings.add_settings(general_settings)
        data = DataHandler(settings)

        print('####################################################################################################')
        print(f'EXP: ', settings.data.dataset)
        print(f'SEED : ', seed)

        for curr_method in methods:

            # print(f'method: ', curr_method)

            if curr_method == 'ITL':

                lambda_par_range = [1]
                model = FixedMahalanobis(data, np.zeros(data.features_tr[0].shape[1]), np.eye(data.features_tr[0].shape[1]), lambda_par_range, loss_name)

            elif curr_method == 'unconditional_Bias':

                lambda_par_range = [1]
                gamma_bias_uncond_par_range = [10 ** i for i in np.linspace(-7, 7, 14)]
                gamma_bias_cond_par_range = [0]
                gamma_feature_par_range = [0]
                # par_range = product(lambda_par_range, gamma_bias_uncond_par_range, gamma_bias_cond_par_range, gamma_feature_uncond_par_range, gamma_feature_cond_par_range)
                par_range = [(a, b, c, d, d) for a in lambda_par_range for b in gamma_bias_uncond_par_range for c in gamma_bias_cond_par_range for d in gamma_feature_par_range]

                model = MetaLearningMahalanobis(data, par_range, loss_name, feature_map_name_bias, feature_map_name_feature, r, W, settings.data.dataset)

            elif curr_method == 'unconditional_Feature':

                lambda_par_range = [1]
                gamma_bias_par_range = [0]
                gamma_feature_uncond_par_range = [10 ** i for i in np.linspace(-7, 7, 14)]
                gamma_feature_cond_par_range = [0]
                # par_range = product(lambda_par_range, gamma_bias_uncond_par_range, gamma_bias_cond_par_range, gamma_feature_uncond_par_range, gamma_feature_cond_par_range)
                par_range = [(a, b, b, c, d) for a in lambda_par_range for b in gamma_bias_par_range for c in gamma_feature_uncond_par_range for d in gamma_feature_cond_par_range]

                model = MetaLearningMahalanobis(data, par_range, loss_name, feature_map_name_bias, feature_map_name_feature, r, W, settings.data.dataset)

            elif curr_method == 'unconditional_Mahalanobis':

                lambda_par_range = [1]
                gamma_uncond_par_range = [10 ** i for i in np.linspace(-7, 7, 14)]
                gamma_cond_par_range = [0]
                # par_range = product(lambda_par_range, gamma_bias_uncond_par_range, gamma_bias_cond_par_range, gamma_feature_uncond_par_range, gamma_feature_cond_par_range)
                par_range = [(a, b, c, b, c) for a in lambda_par_range for b in gamma_uncond_par_range for c in gamma_cond_par_range]

                model = MetaLearningMahalanobis(data, par_range, loss_name, feature_map_name_bias, feature_map_name_feature, r, W, settings.data.dataset)

            elif curr_method == 'conditional_Bias':

                lambda_par_range = [1]
                gamma_par_bias_range = [10 ** i for i in np.linspace(-7, 7, 14)]
                gamma_par_feature_range = [0]

                # par_range = product(lambda_par_range, gamma_bias_uncond_par_range, gamma_bias_cond_par_range, gamma_feature_uncond_par_range, gamma_feature_cond_par_range)
                par_range = [(a, b, b, c, c) for a in lambda_par_range for b in gamma_par_bias_range for c in gamma_par_feature_range]

                model = MetaLearningMahalanobis(data, par_range, loss_name, feature_map_name_bias, feature_map_name_feature, r, W, settings.data.dataset)

            elif curr_method == 'conditional_Feature':

                lambda_par_range = [1]
                gamma_par_bias_range = [0]
                gamma_par_feature_range = [10 ** i for i in np.linspace(-7, 7, 14)]

                par_range = [(a, b, b, c, c) for a in lambda_par_range for b in gamma_par_bias_range for c in gamma_par_feature_range]

                model = MetaLearningMahalanobis(data, par_range, loss_name, feature_map_name_bias, feature_map_name_feature, r, W, settings.data.dataset)

            elif curr_method == 'conditional_Mahalanobis':

                lambda_par_range = [1]
                gamma_par_range = [10 ** i for i in np.linspace(-7, 7, 14)]

                par_range = [(a, b, b, b, b) for a in lambda_par_range for b in gamma_par_range]

                model = MetaLearningMahalanobis(data, par_range, loss_name, feature_map_name_bias, feature_map_name_feature, r, W, settings.data.dataset)

            errors = model.fit()
            results[curr_method].append(errors)

            print('%s DONE %5.2f' % (curr_method, time.time() - tt))
            print('--------------------------------------------------------------------------------------')

        print('COMPLETE SEED DONE: %d | %5.2f sec' % (seed, time.time() - tt))

        print('####################################################################################################')

    np.save(settings.data.dataset + '_' + 'temp_test_error' + '_' + str(datetime.datetime.now()).replace(':', '') + '.npy', results)
    plot_stuff(results, methods, settings.data.dataset)

    # # to load the saved results
    # read_dictionary = np.load('my_file.npy', allow_pickle='TRUE').item()
    # print(read_dictionary['hello'])  # displays "world"

    exit()


if __name__ == "__main__":

    main()
