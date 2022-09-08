
import numpy as np
import scipy


# loss function's subgradient
def subgradient(x, y, w, loss_name=None):

    # absolute loss subgradient
    if loss_name == 'absolute':
        pred = x @ w
        if y < pred:
            return x, 1
        elif y >= pred:
            return - x, -1
        raise ValueError("Unknown loss.")


# loss function
def loss(x, y, w, loss_name=None):

    # absolute loss
    if loss_name == 'absolute':
        pred = x @ w
        if hasattr(y, '__len__'):
            return 1 / len(y) * np.sum(np.abs(y - pred))
        else:
            return np.abs(y - pred)
    else:
        raise ValueError("Unknown loss")


# feature map for the conditional datasets
def feature_map(x, y, feature_map_name=None, r=None, W=None):

    if feature_map_name == 'linear_with_labels':
        if y is None:
            x_transformed = np.mean(x, axis=0)
            x_transformed = np.append(x_transformed, [1.])
        else:
            n_points, n_dims = x.shape
            x_transformed = np.zeros(2 * n_dims)
            for idx_n in range(n_points):
                if np.shape(x[idx_n, :])[0] == 1:
                   x_curr = x[idx_n, :].toarray().ravel()
                else:
                   x_curr = x[idx_n, :]
                x_tmp = np.tensordot(x_curr, [y[idx_n], 1.], 0)
                x_transformed = x_transformed + np.ravel(x_tmp)
            x_transformed = x_transformed / n_points
            x_transformed = np.append(x_transformed, [1.])
    elif feature_map_name == 'ls_regressor':
        inner_lambda = 1
        x_new = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        x_transformed = np.linalg.solve(x_new.T @ x_new + inner_lambda * np.eye(x_new.shape[1]), x_new.T @ y)
    elif feature_map_name == 'recommenders':
        if y is None:
            x_transformed = np.mean(x, axis=0)
            x_transformed = np.append(x_transformed, [1.])
        else:
            max_score = r
            radial_step = 2 * np.pi / (4 * max_score)

            n_points, n_dims = x.shape
            x_transformed = np.zeros(n_dims)
            for idx_n in range(n_points):
                if np.shape(x[idx_n, :])[0] == 1:
                    x_curr = x[idx_n, :].toarray().ravel()
                else:
                    x_curr = x[idx_n, :]
                x_curr = x_curr * y[idx_n]
                x_transformed = x_transformed + np.ravel(x_curr)

            x_transformed = np.array([np.cos(radial_step * x_transformed) * (x_transformed != 0),
                                      np.sin(radial_step * x_transformed) * (x_transformed != 0)])
            x_transformed = x_transformed.ravel() / n_points
            x_transformed = np.append(x_transformed, [1.])
    else:
        raise ValueError("Unknown feature map.")

    return x_transformed


# projection on positive semidefinite matrices
def proj(matrix_d):

    s, matrix_u = scipy.linalg.eigh(matrix_d)
    s = np.maximum(s, 0)
    return matrix_u @ np.diag(s) @ matrix_u.T


