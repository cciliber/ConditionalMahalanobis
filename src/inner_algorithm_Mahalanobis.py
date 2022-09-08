
import numpy as np
import scipy.sparse
from src.general_functions import subgradient, loss
import matplotlib.pyplot as plt


def inner_algorithm_Mahalanobis(x, y, lambda_par, bias, feature, loss_name):

    # curr_meta_parameter_inv = np.linalg.pinv(curr_meta_parameter)

    epochs_number = 1

    n_points, n_dims = x.shape
    k = 0  # total number of iterations
    epochs_number_temp = 1
    average_cum_error = 0
    temp_weight_vectors = []
    temp_gradients = []
    all_individual_losses = []
    all_average_cum_err = []
    dual_vector = []

    # curr_weights = np.zeros(n_dims)
    # curr_weights = np.zeros(x.shape[1])
    curr_weights = bias

    shuffled_indexes = list(range(n_points))
    # np.random.shuffle(shuffled_indexes)

    while epochs_number_temp <= epochs_number:

        for inner_iteration, curr_point_idx in enumerate(shuffled_indexes):

            # print(inner_iteration)

            if inner_iteration == n_points - 1:
                epochs_number_temp = epochs_number_temp + 1
                # print('END EPOCH')

            k = k + 1

            # receive a new datapoint
            curr_x = x[curr_point_idx, :]
            curr_y = y[curr_point_idx]

            if type(curr_x) == scipy.sparse.csc.csc_matrix:
                curr_x = np.transpose(curr_x).toarray().ravel()

            loss_current = loss(curr_x, curr_y, curr_weights, loss_name)
            all_individual_losses.append(loss_current)
            average_cum_error = (1 / k) * ((k - 1) * average_cum_error + loss_current)
            all_average_cum_err.append(average_cum_error)

            # compute the gradient of the regularized loss
            subgrad, subgrad_scal = subgradient(curr_x, curr_y, curr_weights, loss_name)
            # full_gradient = curr_meta_parameter @ (subgrad + lambda_par * curr_meta_parameter_inv @ curr_weights)
            full_gradient = feature @ subgrad + lambda_par * curr_weights
            temp_gradients.append(subgrad)
            dual_vector.append(subgrad_scal)

            # update the inner weight vector
            curr_weights = curr_weights - (1/(lambda_par * k)) * full_gradient
            temp_weight_vectors.append(curr_weights)

    # # plot the average cum_error across the iteration k
    # plt.plot(all_average_cum_err)
    # plt.title('Instantaneous Average Cumulative Error')
    # # # plt.ylim(top=12, bottom=0)
    # plt.pause(0.5)
    # print(average_cum_error)

    # print('last average cumulative error: %10.3f' % average_cum_error)

    average_weights = np.mean(temp_weight_vectors, axis=0)

    return curr_weights, average_weights, dual_vector
