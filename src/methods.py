
import numpy as np
from src.general_functions import loss, feature_map, proj
from src.inner_algorithm_Mahalanobis import inner_algorithm_Mahalanobis
# from itertools import product
import multiprocessing
import time


class FixedMahalanobis:

    def __init__(self, data, fixed_bias, fixed_feature, lambda_par_range, loss_name):

        self.fixed_bias = fixed_bias
        self.fixed_feature = fixed_feature
        self.lambda_par_range = lambda_par_range
        self.loss_name = loss_name
        self.data = data

    def process_FixedMahalanobis(self, lambda_idx):

        lambda_par = self.lambda_par_range[lambda_idx]
        data = self.data

        # computing the average test error on the validation tasks
        all_validation_errors = []

        for _, task_val in enumerate(data.val_task_indexes):
            x_tr = data.features_tr[task_val]
            y_tr = data.labels_tr[task_val]
            x_ts = data.features_ts[task_val]
            y_ts = data.labels_ts[task_val]

            curr_weights, average_weights, dual_vector = inner_algorithm_Mahalanobis(x_tr, y_tr, lambda_par, self.fixed_bias, self.fixed_feature, self.loss_name)
            validation_error = loss(x_ts, y_ts, average_weights, self.loss_name)
            all_validation_errors.append(validation_error)

        average_validation_error = np.mean(all_validation_errors)

        return average_validation_error, lambda_par

    def fit(self):

        data = self.data
        # we use the same lambda for each task
        num_cores = multiprocessing.cpu_count()
        # results_validation = Parallel(n_jobs=num_cores)(
        #     delayed(self.process_FixedFeature)(data, lambda_idx) for lambda_idx in range(len(self.lambda_par_range)))
        with multiprocessing.Pool(num_cores) as pool:
            results_validation = pool.map(self.process_FixedMahalanobis, range(len(self.lambda_par_range)))

        results_to_select_min = []
        for idx in range(len(self.lambda_par_range)):
            results_to_select_min.append(results_validation[idx][0])
        results_to_select_min = np.asarray(results_to_select_min)
        best_indexes = results_to_select_min.argmin()
        best_perf, best_lambda_par = results_validation[best_indexes]

        all_test_errors = []

        for _, task_ts in enumerate(data.test_task_indexes):

            x_tr = data.features_tr[task_ts]
            y_tr = data.labels_tr[task_ts]
            x_ts = data.features_ts[task_ts]
            y_ts = data.labels_ts[task_ts]

            curr_weights, average_weights, dual_vector = inner_algorithm_Mahalanobis(x_tr, y_tr, best_lambda_par, self.fixed_bias, self.fixed_feature, self.loss_name)
            test_error = loss(x_ts, y_ts, average_weights, self.loss_name)
            all_test_errors.append(test_error)

        average_test_error = np.mean(all_test_errors)
        all_best_performances = average_test_error * np.ones(len(data.tr_task_indexes))

        print(f'best lambda: ', best_lambda_par)
        print(f'best test error: ', all_best_performances[- 1])
        # plt.plot(all_validation_errors)
        # plt.title('Validation curve')
        # # # plt.ylim(top=12, bottom=0)
        # plt.pause(0.5)

        return all_best_performances


class MetaLearningMahalanobis:

    def __init__(self, data, par_range, loss_name, feature_map_name_bias, feature_map_name_feature, r, W, dataset):

        self.par_range = par_range
        self.loss_name = loss_name
        self.feature_map_name_bias = feature_map_name_bias
        self.feature_map_name_feature = feature_map_name_feature
        self.r = r
        self.W = W
        self.dataset = dataset
        self.data = data

    def process_conditional_Mahalanobis(self, idx):

        (lambda_par, gamma_par_bias_uncond, gamma_par_bias_cond, gamma_par_feature_uncond, gamma_par_feature_cond) = self.par_range[idx]
        data = self.data

        all_average_val_errors_temp = []  # temporary memory for the best val error curve
        all_average_test_errors_temp = []  # temporary memory for the best test error curve

        # initialize meta-parameter
        curr_bias = np.zeros(data.features_tr[0].shape[1])
        curr_feature = np.eye(data.features_tr[0].shape[1])

        if gamma_par_bias_cond != 0:
            test_for_shape_bias = feature_map(data.features_tr[0], data.labels_tr[0], self.feature_map_name_bias, self.r, self.W)
            curr_A = np.zeros([data.features_tr[0].shape[1], test_for_shape_bias.shape[0]])
            sum_A = np.zeros([data.features_tr[0].shape[1], test_for_shape_bias.shape[0]])
            avg_A = np.zeros([data.features_tr[0].shape[1], test_for_shape_bias.shape[0]])
        if gamma_par_bias_uncond != 0:
            curr_b = np.zeros(data.features_tr[0].shape[1])
            sum_b = np.zeros(data.features_tr[0].shape[1])
            avg_b = np.zeros(data.features_tr[0].shape[1])
        if gamma_par_feature_cond != 0:
            test_for_shape_feature = feature_map(data.features_tr[0], data.labels_tr[0], self.feature_map_name_feature, self.r, self.W)
            curr_C = np.zeros([data.features_tr[0].shape[1] * test_for_shape_feature.shape[0], data.features_tr[0].shape[1] * test_for_shape_feature.shape[0]])
            sum_C = np.zeros([data.features_tr[0].shape[1] * test_for_shape_feature.shape[0], data.features_tr[0].shape[1] * test_for_shape_feature.shape[0]])
            avg_C = np.zeros([data.features_tr[0].shape[1] * test_for_shape_feature.shape[0], data.features_tr[0].shape[1] * test_for_shape_feature.shape[0]])
        if gamma_par_feature_uncond != 0:
            curr_D = np.eye(data.features_tr[0].shape[1])
            sum_D = np.eye(data.features_tr[0].shape[1])
            avg_D = np.eye(data.features_tr[0].shape[1])

        idx_avg = 1

        for task_tr_index, task_tr in enumerate(data.tr_task_indexes):

            tt = time.time()

            # print(f'TRAINING task', task_tr_index + 1)
            # t = time.time

            x = data.features_tr[task_tr]
            y = data.labels_tr[task_tr]
            n_points, n_dims = x.shape
            # if self.dataset == 'movies':
            #    s = data.all_side_info[task_tr]
            #    # s = x

            # compute the transformed conditional dataset (we use the training sets as the conditional sets)
            # if self.dataset == 'movies':
            #     x_trasf_feature = feature_map(s, y, self.feature_map_name, self.r, self.W)
            # else:
            #     x_trasf_feature = feature_map(x, y, self.feature_map_name, self.r, self.W)

            # update the meta-parameter
            if gamma_par_bias_uncond != 0:  # and gamma_par_bias_cond == 0:
                curr_bias = avg_b
            if gamma_par_bias_cond != 0:
                x_trasf_feature_bias = feature_map(x, y, self.feature_map_name_bias, self.r, self.W)
                curr_bias = curr_bias + avg_A @ x_trasf_feature_bias
            if gamma_par_feature_uncond != 0:  # and gamma_par_feature_cond == 0:
                curr_feature = avg_D
            if gamma_par_feature_cond != 0:
                x_trasf_feature_feature = feature_map(x, y, self.feature_map_name_feature, self.r, self.W)
                temp_matrix = np.tensordot(np.eye(data.features_tr[0].shape[1]), x_trasf_feature_feature, 0)
                temp_matrix_reshape = np.reshape(temp_matrix, (data.features_tr[0].shape[1] * test_for_shape_feature.shape[0], data.features_tr[0].shape[1]))
                curr_feature = curr_feature + temp_matrix_reshape.T @ avg_C @ temp_matrix_reshape

            curr_weights, average_weights, dual_vector = inner_algorithm_Mahalanobis(x, y, lambda_par, curr_bias, curr_feature, self.loss_name)

            # compute the meta-gradient
            if self.loss_name == 'hinge_multi':
                # GD: FIX THIS
                meta_gradient_b = - (1 / (2 * lambda_par * (n_points ** 2))) * dual_vector @ dual_vector.T
            else:
                if gamma_par_bias_uncond != 0:
                    meta_gradient_b = lambda_par * (1 / n_points) * x.T @ dual_vector
                if gamma_par_bias_cond != 0:
                    meta_gradient_A = np.tensordot(meta_gradient_b, x_trasf_feature_bias, 0)
                if gamma_par_feature_uncond != 0:
                    meta_gradient_b = lambda_par * (1 / n_points) * x.T @ dual_vector
                    meta_gradient_D = - (1 / (2 * lambda_par * (n_points ** 2))) * np.tensordot(meta_gradient_b, meta_gradient_b, 0) + (2 / (lambda_par * (n_points ** 2))) * x.T @ x
                if gamma_par_feature_cond != 0:
                    meta_gradient_C = temp_matrix_reshape @ meta_gradient_D @ temp_matrix_reshape.T

            # update the meta_parameter
            if gamma_par_bias_cond != 0:
                curr_A = curr_A - gamma_par_bias_cond * meta_gradient_A
            if gamma_par_bias_uncond != 0:
                curr_b = curr_b - gamma_par_bias_uncond * meta_gradient_b
            if gamma_par_feature_cond != 0:
                curr_C = proj(curr_C - gamma_par_feature_cond * meta_gradient_C)
            if gamma_par_feature_uncond != 0:
                curr_D = proj(curr_D - gamma_par_feature_uncond * meta_gradient_D)

            if gamma_par_bias_cond != 0:
                sum_A = sum_A + curr_A
                avg_A = sum_A / idx_avg
            if gamma_par_bias_uncond != 0:
                sum_b = sum_b + curr_b
                avg_b = sum_b / idx_avg
            if gamma_par_feature_cond != 0:
                sum_C = sum_C + curr_C
                avg_C = sum_C / idx_avg
            if gamma_par_feature_uncond != 0:
                sum_D = sum_D + curr_D
                avg_D = sum_D / idx_avg

            idx_avg = idx_avg + 1

            # print('one iteration done in ', time.time() - tt)

            # all_meta_parameters_temp.append(curr_meta_parameter)
            # average_meta_parameter = np.mean(all_meta_parameters_temp, axis=0)

            # compute the error on the validation and test tasks with average_meta_parameter
            all_val_errors_temp = []
            for _, task_val in enumerate(data.val_task_indexes):
                x_tr = data.features_tr[task_val]
                y_tr = data.labels_tr[task_val]
                x_ts = data.features_ts[task_val]
                y_ts = data.labels_ts[task_val]
                # if self.dataset == 'movies':
                #     s = data.all_side_info[task_val]
                #     # s = x_tr

                # compute the transformed conditional dataset (we use the training sets as the conditional sets)
                # if self.dataset == 'movies':
                #     x_trasf_feature = feature_map(s, y_tr, self.feature_map_name, self.r, self.W)
                # else:
                #     x_trasf_feature = feature_map(x_tr, y_tr, self.feature_map_name, self.r, self.W)
                if gamma_par_bias_uncond != 0:
                    curr_bias = avg_b
                if gamma_par_bias_cond != 0:
                    x_trasf_feature_bias = feature_map(x_tr, y_tr, self.feature_map_name_bias, self.r, self.W)
                    curr_bias = curr_bias + avg_A @ x_trasf_feature_bias
                if gamma_par_feature_uncond != 0:
                    curr_feature = avg_D
                if gamma_par_feature_cond != 0:
                    x_trasf_feature_feature = feature_map(x_tr, y_tr, self.feature_map_name_feature, self.r, self.W)
                    temp_matrix = np.tensordot(np.eye(data.features_tr[0].shape[1]), x_trasf_feature_feature, 0)
                    temp_matrix_reshape = np.reshape(temp_matrix, (data.features_tr[0].shape[1] * test_for_shape_feature.shape[0], data.features_tr[0].shape[1]))
                    curr_feature = curr_feature + temp_matrix_reshape.T @ avg_C @ temp_matrix_reshape

                curr_weights, average_weights, dual_vector = inner_algorithm_Mahalanobis(x_tr, y_tr, lambda_par, curr_bias, curr_feature, self.loss_name)

                val_error = loss(x_ts, y_ts, average_weights, self.loss_name)
                all_val_errors_temp.append(val_error)

            average_val_error = np.mean(all_val_errors_temp)
            all_average_val_errors_temp.append(average_val_error)

            all_test_errors_temp = []
            for _, task_ts in enumerate(data.test_task_indexes):
                x_tr = data.features_tr[task_ts]
                y_tr = data.labels_tr[task_ts]
                x_ts = data.features_ts[task_ts]
                y_ts = data.labels_ts[task_ts]
                # if self.dataset == 'movies':
                #     s = data.all_side_info[task_ts]
                #     # s = x_tr

                # compute the transformed conditional dataset (we use the training sets as the conditional sets)
                # if self.dataset == 'movies':
                #     x_trasf_feature = feature_map(s, y_tr, self.feature_map_name, self.r, self.W)
                # else:
                #     x_trasf_feature = feature_map(x_tr, y_tr, self.feature_map_name, self.r, self.W)
                if gamma_par_bias_uncond != 0:
                    curr_bias = avg_b
                if gamma_par_bias_cond != 0:
                    x_trasf_feature_bias = feature_map(x_tr, y_tr, self.feature_map_name_bias, self.r, self.W)
                    x_trasf_feature_feature = feature_map(x_tr, y_tr, self.feature_map_name_feature, self.r, self.W)
                    curr_bias = curr_bias + avg_A @ x_trasf_feature_bias
                if gamma_par_feature_uncond != 0:
                    curr_feature = avg_D
                if gamma_par_feature_cond != 0:
                    temp_matrix = np.tensordot(np.eye(data.features_tr[0].shape[1]), x_trasf_feature_feature, 0)
                    temp_matrix_reshape = np.reshape(temp_matrix, (data.features_tr[0].shape[1] * test_for_shape_feature.shape[0], data.features_tr[0].shape[1]))
                    curr_feature = curr_feature + temp_matrix_reshape.T @ avg_C @ temp_matrix_reshape

                curr_weights, average_weights, dual_vector = inner_algorithm_Mahalanobis(x_tr, y_tr, lambda_par, curr_bias, curr_feature, self.loss_name)
                test_error = loss(x_ts, y_ts, average_weights, self.loss_name)
                all_test_errors_temp.append(test_error)
            average_test_error = np.mean(all_test_errors_temp)
            all_average_test_errors_temp.append(average_test_error)

        return average_val_error, lambda_par, gamma_par_bias_uncond, gamma_par_bias_cond, gamma_par_feature_uncond, gamma_par_feature_cond, all_average_test_errors_temp

    def fit(self):

        data = self.data
        num_cores = multiprocessing.cpu_count()
        set_of_indexes = range(len(self.par_range))
        # results_validation = Parallel(n_jobs=num_cores)(
        #     delayed(self.process_conditional)(data, lambda_idx, gamma_idx)
        #     for lambda_idx, gamma_idx in product(range(len(self.lambda_par_range)), range(len(self.gamma_par_range))))
        with multiprocessing.Pool(num_cores) as pool:
            results_validation = pool.map(self.process_conditional_Mahalanobis, set_of_indexes)

        results_to_select_min = []
        for idx in set_of_indexes:
            results_to_select_min.append(results_validation[idx][0])

        results_to_select_min = np.asarray(results_to_select_min)
        best_indexes = results_to_select_min.argmin()
        best_perf, best_lambda_par, best_gamma_par_bias_uncond, best_gamma_par_bias_cond, best_gamma_par_feature_uncond, best_gamma_par_feature_cond, all_best_performances = results_validation[best_indexes]

        # plt.plot(all_best_performances)
        # plt.title('best lambda ' + str(best_lambda_par) + ' | ' + 'best gamma ' + str(best_gamma_par))
        # plt.ylim(top=12, bottom=0)
        # plt.pause(0.1)

        print('--------------------------------------------------------------------------------------')

        print(f'best gamma bias uncond: ', best_gamma_par_bias_uncond)
        print(f'best gamma bias cond: ', best_gamma_par_bias_cond)
        print(f'best gamma feature uncond: ', best_gamma_par_feature_uncond)
        print(f'best gamma feature uncond: ', best_gamma_par_feature_cond)
        print(f'best lambda: ', best_lambda_par)
        print(f'best test error: ', all_best_performances[- 1])

        return all_best_performances


