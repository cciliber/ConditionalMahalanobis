
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from scipy import io as sio
import pandas as pd
from fiblat import sphere_lattice


class Settings:

    def __init__(self, dictionary, struct_name=None):
        if struct_name is None:
            self.__dict__.update(**dictionary)
        else:
            temp_settings = Settings(dictionary)
            setattr(self, struct_name, temp_settings)

    def add_settings(self, dictionary, struct_name=None):

        if struct_name is None:
            self.__dict__.update(dictionary)
        else:
            if hasattr(self, struct_name):
                temp_settings = getattr(self, struct_name)
                temp_settings.__dict__.update(dictionary)
            else:
                temp_settings = Settings(dictionary)
            setattr(self, struct_name, temp_settings)


class DataHandler:

    def __init__(self, settings):
        settings.add_settings({'n_all_tasks': settings.data.n_tr_tasks + settings.data.n_val_tasks + settings.data.n_test_tasks}, 'data')
        self.settings = settings
        self.features_tr = [None] * settings.data.n_all_tasks
        self.features_ts = [None] * settings.data.n_all_tasks
        self.labels_tr = [None] * settings.data.n_all_tasks
        self.labels_ts = [None] * settings.data.n_all_tasks
        self.all_side_info = [None] * settings.data.n_all_tasks
        self.oracle_unconditional = None
        self.oracle_conditional = None

        self.tr_task_indexes = None
        self.val_task_indexes = None
        self.test_task_indexes = None

        if self.settings.data.dataset == 'synthetic_feature_CLUSTERS_gen_old_bias_paper':
            self.synthetic_feature_CLUSTERS_gen_old_bias_paper()
        elif self.settings.data.dataset == 'synthetic-regression-3-CLUSTERS-SMALL':
            self.synthetic_feature_CLUSTERS_gen()
        elif self.settings.data.dataset == 'synthetic-regression-3-CLUSTERS-LARGE':
            self.synthetic_feature_CLUSTERS_gen()
        elif self.settings.data.dataset == 'lenk':
            self.lenk_data_gen()
        elif self.settings.data.dataset == 'movies':
            self.movielens_gen()
        elif self.settings.data.dataset == 'jester':
            self.jester_gen()
        else:
            raise ValueError('Invalid dataset')

    def synthetic_feature_CLUSTERS_gen_old_bias_paper(self):

        number_clusters = 2
        n_tasks = self.settings.data.n_all_tasks
        clusters_belonging_indexes = np.random.randint(number_clusters, size=(1, n_tasks))

        translation_centroids_weights = 8
        zero_centroid = np.zeros((1, self.settings.data.n_dims))
        other_centroid = translation_centroids_weights * np.ones((1, self.settings.data.n_dims))
        one_centroid = np.ones((1, self.settings.data.n_dims))

        input_centroids = np.concatenate((-one_centroid, one_centroid), axis=0)

        # put the two centroids as cols of a matrix
        all_centroids_weights = np.concatenate((zero_centroid, other_centroid), axis=0)
        matrix_w = np.zeros((self.settings.data.n_dims, self.settings.data.n_all_tasks))

        for task_idx in range(self.settings.data.n_all_tasks):

            cluster_idx = clusters_belonging_indexes[0, task_idx]

            centroid_weights = all_centroids_weights[cluster_idx, :]
            centroid_features = input_centroids[cluster_idx, :]

            # generate the dataset as a matrix with rows as samples and columns as features
            # with data generated as a gaussian around the input_centroid
            features = centroid_features + np.random.normal(0, 1, (self.settings.data.n_all_points, self.settings.data.n_dims))

            # generating and normalizing the weight vectors
            weight_vector = centroid_weights + np.random.normal(loc=np.zeros(self.settings.data.n_dims),
                                                                scale=1).ravel()

            matrix_w[:, task_idx] = weight_vector

            # generating labels and adding noise
            clean_labels = features @ weight_vector
            signal_to_noise_ratio = 1
            standard_noise = np.random.randn(self.settings.data.n_all_points)
            noise_std = np.sqrt(np.var(clean_labels) / (signal_to_noise_ratio * np.var(standard_noise)))
            noisy_labels = clean_labels + noise_std * standard_noise

            # split into training and test
            tr_indexes, ts_indexes = train_test_split(np.arange(0, self.settings.data.n_all_points),
                                                      test_size=self.settings.data.ts_points_pct)
            features_tr = features[tr_indexes]
            labels_tr = noisy_labels[tr_indexes]
            features_ts = features[ts_indexes]
            labels_ts = noisy_labels[ts_indexes]

            self.features_tr[task_idx] = features_tr
            self.features_ts[task_idx] = features_ts
            self.labels_tr[task_idx] = labels_tr
            self.labels_ts[task_idx] = labels_ts

        self.tr_task_indexes = np.arange(0, self.settings.data.n_tr_tasks)
        self.val_task_indexes = np.arange(self.settings.data.n_tr_tasks,
                                          self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks)
        self.test_task_indexes = np.arange(self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks,
                                           self.settings.data.n_all_tasks)



    def synthetic_feature_CLUSTERS_gen(self):

        number_clusters = self.settings.data.number_clusters
        sparsity = self.settings.data.sparsity

        n_tasks = self.settings.data.n_all_tasks
        clusters_belonging_indexes = np.random.randint(number_clusters, size=(1, n_tasks))

        all_sparsity = int(sparsity * number_clusters)
        all_fixed_sparsity_vec = np.random.choice(np.arange(0, self.settings.data.n_dims), all_sparsity, replace=False)
        all_fixed_sparsity = np.zeros((number_clusters, sparsity))
        all_fixed_sparsity = all_fixed_sparsity.astype(int)

        for cluster_idx in range(number_clusters):
            all_fixed_sparsity[cluster_idx, :] = all_fixed_sparsity_vec[cluster_idx * sparsity:cluster_idx * sparsity + sparsity]

        translation_centroids_weights = 10
        translation_centroids_weights = translation_centroids_weights * np.ones(self.settings.data.n_dims)
        sphere_radius_weights = 3
        # translation_centroids_inputs = 1 * np.ones(self.settings.data.n_dims)
        # radius_inputs = 2

        all_centroids_weights = sphere_radius_weights * sphere_lattice(self.settings.data.n_dims, number_clusters)
        # all_centroids_inputs = radius_inputs * sphere_lattice(self.settings.data.n_dims, number_clusters)

        matrix_w = np.zeros((self.settings.data.n_dims, self.settings.data.n_all_tasks))

        for task_idx in range(self.settings.data.n_all_tasks):

            cluster_idx = clusters_belonging_indexes[0, task_idx]
            fixed_sparsity = all_fixed_sparsity[cluster_idx, :]

            # centroid_inputs = translation_centroids_inputs + all_centroids_inputs[cluster_idx, :]
            centroid_weights = translation_centroids_weights + all_centroids_weights[cluster_idx, :]

            # generating and normalizing the inputs
            # features = np.random.randn(self.settings.data.n_all_points, self.settings.data.n_dims)
            features = np.zeros((self.settings.data.n_all_points, self.settings.data.n_dims))
            for idx in range(self.settings.data.n_all_points):
                features[idx, fixed_sparsity] = np.random.randn(sparsity, 1).ravel()
                features[idx, :] = (features[idx, :] / norm(features[idx, :])).ravel()  # * np.random.randint(1, 10)
                # features[idx, :] = features + centroid_inputs

            # generating and normalizing the weight vectors
            weight_vector = np.zeros((self.settings.data.n_dims, 1))
            weight_vector[fixed_sparsity] = np.random.randn(sparsity, 1)
            weight_vector = (weight_vector / norm(weight_vector)).ravel()  # * np.random.randint(1, 10)

            single_weight_radius = 0.5
            weight_vector = single_weight_radius * weight_vector
            
            weight_vector = weight_vector + centroid_weights  # + np.random.normal(loc=np.zeros(self.settings.data.n_dims),scale=1).ravel()

            matrix_w[:, task_idx] = weight_vector

            # generating labels and adding noise
            clean_labels = features @ weight_vector
            noise_std = self.settings.data.noise_std
            noisy_labels = clean_labels + noise_std * np.random.randn(self.settings.data.n_all_points)

            # split into training and test
            tr_indexes, ts_indexes = train_test_split(np.arange(0, self.settings.data.n_all_points),
                                                      test_size=self.settings.data.ts_points_pct)
            features_tr = features[tr_indexes]
            labels_tr = noisy_labels[tr_indexes]
            features_ts = features[ts_indexes]
            labels_ts = noisy_labels[ts_indexes]

            self.features_tr[task_idx] = features_tr
            self.features_ts[task_idx] = features_ts
            self.labels_tr[task_idx] = labels_tr
            self.labels_ts[task_idx] = labels_ts

        self.tr_task_indexes = np.arange(0, self.settings.data.n_tr_tasks)
        self.val_task_indexes = np.arange(self.settings.data.n_tr_tasks,
                                          self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks)
        self.test_task_indexes = np.arange(self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks,
                                           self.settings.data.n_all_tasks)

    def lenk_data_gen(self):

        temp = sio.loadmat('data/lenk_data.mat')
        train_data = temp['Traindata']  # 2880x15  last feature is output (score from 0 to 10) (144 tasks of 20 elements)
        test_data = temp['Testdata']  # 720x15 last feature is y (score from 0 to 10) (26 tasks of 20 elements)

        Y = train_data[:, 14]
        Y_test = test_data[:, 14]
        X = train_data[:, :14]
        X_test = test_data[:, :14]

        n_tasks = 170  # --> n_tot_tasks
        n_tot = 20
        ne_tr = 16  # number of elements on train set per task
        ne_test = 4  # number of elements on test set per task

        def split_tasks(data, nt, number_of_elements):
            return [data[i * number_of_elements:(i + 1) * number_of_elements] for i in range(nt)]

        data_m = split_tasks(X, n_tasks, ne_tr)
        labels_m = split_tasks(Y, n_tasks, ne_tr)

        data_test_m = split_tasks(X_test, n_tasks, ne_test)
        labels_test_m = split_tasks(Y_test, n_tasks, ne_test)

        # shuffled_tasks = np.random.permutation(n_tasks)
        shuffled_tasks = list(range(self.settings.data.n_all_tasks))
        np.random.shuffle(shuffled_tasks)

        for task_idx, task in enumerate(shuffled_tasks):

            es = np.random.permutation(len(labels_m[task_idx]))
            # es = list(range(len(labels_m[task_idx])))

            X_train, Y_train = data_m[task_idx][es], labels_m[task_idx][es]
            X_test, Y_test = data_test_m[task_idx], labels_test_m[task_idx]

            Y_train = Y_train.ravel()
            X_train = X_train

            self.features_tr[task_idx] = X_train
            self.features_ts[task_idx] = X_test
            self.labels_tr[task_idx] = Y_train
            self.labels_ts[task_idx] = Y_test

        self.tr_task_indexes = np.arange(0, self.settings.data.n_tr_tasks)
        self.val_task_indexes = np.arange(self.settings.data.n_tr_tasks,
                                          self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks)
        self.test_task_indexes = np.arange(self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks,
                                           self.settings.data.n_all_tasks)

    def movielens_gen(self):

        import copy
        import scipy.sparse
        import scipy.io as sio
        from scipy.sparse import csc_matrix

        temp = sio.loadmat('data/ml100kSparse.mat')
        # temp = sio.loadmat('data/ml1mSparse.mat')
        full_matrix = temp['fullMatrix'].astype(float)

        # loading users' side info
        u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']  # pass in column names for each CSV
        users = pd.read_csv('data/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')
        # normalizing the users
        # side_info_users = normalize(pd.get_dummies(users[['age', 'sex', 'occupation']].fillna(0)))
        # side_info_users = pd.get_dummies(users[['age', 'sex', 'occupation']].fillna(0))
        side_info_users = pd.get_dummies(users[['age', 'sex', 'occupation', 'zip_code']].fillna(0))
        side_info_users_matrix = side_info_users.values

        # possibly a parameter?
        top_movies = 20
        minimum_votes_for_top_movies = 10

        # count the number of appearances (i.e. times it was voted) of each movie
        votes_counter = np.sum(full_matrix != 0, axis=0)
        votes_counter = np.ravel(votes_counter)

        # get the most voted movies
        most_voted_idx = np.argsort(-votes_counter)[:top_movies]

        # get the users that have voted a minimum number of movies in the top list
        users_top_votes = np.sum(full_matrix[:, most_voted_idx] != 0, axis=1)
        keep_user_idx = np.nonzero(users_top_votes >= minimum_votes_for_top_movies)[0]

        # ---- DEPRECATED ----
        # # count the number each movie appears in the dataset and remove those that are too rare
        # columns_to_keep = []
        # number_of_appearences_all_films = []  # Added in order to consider only the most voted films
        # for column in range(full_matrix.shape[1]):
        #     number_of_appearences = len(np.nonzero(full_matrix[:, column])[0])
        #     number_of_appearences_all_films.append(number_of_appearences)  # Added to consider only the most voted films
        #     if number_of_appearences >= 20:
        #         columns_to_keep.append(column)
        #
        # full_matrix = full_matrix[:, columns_to_keep]

        full_matrix = full_matrix[:, most_voted_idx]
        full_matrix = full_matrix[keep_user_idx, :]

        n_movies = full_matrix.shape[1]
        self.settings.data.n_dims = n_movies
        n_tot_tasks = full_matrix.shape[0]
        # print(f'n = ', n_movies)
        # print(f'T = ', n_tot_tasks)

        if self.settings.data.n_all_tasks > n_tot_tasks:
            print("################################ WARNING Too Many Training Tasks")
            print("actual_tasks:", n_tot_tasks)
            print("required_tasks:", self.settings.data.n_all_tasks)
            return

        shuffled_task_indexes = np.random.permutation(self.settings.data.n_all_tasks)
        # shuffled_tasks = list(range(self.settings.data.n_all_tasks))
        # np.random.shuffle(shuffled_tasks)

        for task_counter, user in enumerate(shuffled_task_indexes):  # enumerate(shuffled_tasks)

            # loading and normalizing the inputs
            zero_indexes = np.where(full_matrix[user, :].toarray() == 0)[1]
            non_zero_indexes = np.nonzero(full_matrix[user, :])[1]
            features = csc_matrix(np.eye(n_movies))  # np.eye(n_movies)
            features[zero_indexes, zero_indexes] = 0

            # loading the labels
            labels = full_matrix[user, :].toarray().ravel()

            # loading the side info
            side_info = side_info_users_matrix[user, :].ravel()
            self.all_side_info[user] = side_info

            if task_counter >= self.settings.data.n_tr_tasks:

                # split into training and test
                tr_indexes, ts_indexes = train_test_split(non_zero_indexes, test_size=self.settings.data.ts_points_pct)
                features_tr = features[tr_indexes, :]  # copy.deepcopy(features)  (big and not necessary matrices)
                labels_tr = labels[tr_indexes]  # copy.deepcopy(labels)
                # features_tr[ts_indexes, ts_indexes] = 0
                # labels_tr[ts_indexes] = 0

                features_ts = features[ts_indexes, :]  # copy.deepcopy(features)
                labels_ts = labels[ts_indexes]  # copy.deepcopy(labels)
                # features_ts[tr_indexes, tr_indexes] = 0
                # labels_ts[tr_indexes] = 0

                self.features_tr[user] = features_tr
                self.features_ts[user] = features_ts
                self.labels_tr[user] = labels_tr
                self.labels_ts[user] = labels_ts
            else:
                self.features_tr[user] = features[non_zero_indexes, :]  # copy.deepcopy(features)  (big and not necessary matrices)
                self.labels_tr[user] = labels[non_zero_indexes]  # copy.deepcopy(labels)
                # print(type(features_tr[user]))
                # print(type(labels_tr))
                # print(type(features_ts[user]))
                # print(type(labels_ts))

        self.tr_task_indexes = shuffled_task_indexes[:self.settings.data.n_tr_tasks]
        self.val_task_indexes = shuffled_task_indexes[
                                self.settings.data.n_tr_tasks:self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks]
        self.test_task_indexes = shuffled_task_indexes[
                                 self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks:self.settings.data.n_all_tasks]
        self.full_matrix = full_matrix
        self.side_info_users_matrix = side_info_users_matrix

        # self.tr_task_indexes = np.arange(0, self.settings.data.n_tr_tasks)
        # self.val_task_indexes = np.arange(self.settings.data.n_tr_tasks,
        #                                   self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks)
        # self.test_task_indexes = np.arange(self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks,
        #                                    self.settings.data.n_all_tasks)

    def jester_gen(self):

        import copy
        import scipy.io as sio
        from scipy.sparse import csc_matrix
        # temp = sio.loadmat('data/' + self.settings.data.dataset + 'Sparse.mat')
        # full_matrix = temp[self.settings.data.dataset + 'Sparse'].astype(float)
        temp = sio.loadmat('data/jester1Sparse.mat')
        full_matrix = temp['jester1Sparse'].astype(float)
        # rescaling the labels in a such way to get a number between [0, 20] instead of [-10,10]
        full_matrix[full_matrix != 0] = full_matrix[full_matrix != 0] + 11
        # temp = sio.loadmat('data/jester2Sparse.mat')
        # full_matrix = temp['jester2Sparse'].astype(float)
        # temp = sio.loadmat('data/jester3Sparse.mat')
        # full_matrix = temp['jester3Sparse'].astype(float)

        top_jokes = 20
        minimum_votes_for_top_jokes = 10

        # count the number of appearances (i.e. times it was voted) of each movie
        votes_counter = np.sum(full_matrix != 0, axis=0)
        votes_counter = np.ravel(votes_counter)

        # get the most voted movies
        most_voted_idx = np.argsort(-votes_counter)[:top_jokes]

        # get the users that have voted a minimum number of movies in the top list
        users_top_votes = np.sum(full_matrix[:, most_voted_idx] != 0, axis=1)
        keep_user_idx = np.nonzero(users_top_votes >= minimum_votes_for_top_jokes)[0]

        # ---- DEPRECATED ----
        # # count the number each movie appears in the dataset and remove those that are too rare
        # columns_to_keep = []
        # number_of_appearences_all_films = []  # Added in order to consider only the most voted films
        # for column in range(full_matrix.shape[1]):
        #     number_of_appearences = len(np.nonzero(full_matrix[:, column])[0])
        #     number_of_appearences_all_films.append(number_of_appearences)  # Added to consider only the most voted films
        #     if number_of_appearences >= 20:
        #         columns_to_keep.append(column)
        #
        # full_matrix = full_matrix[:, columns_to_keep]

        full_matrix = full_matrix[:, most_voted_idx]
        full_matrix = full_matrix[keep_user_idx, :]

        n_jokes = full_matrix.shape[1]
        self.settings.data.n_dims = n_jokes
        n_tot_tasks = full_matrix.shape[0]
        # print(f'n = ', n_jokes)
        # print(f'T = ', n_tot_tasks)

        if self.settings.data.n_all_tasks > n_tot_tasks:
            print("################################ WARNING Too Many Training Tasks")
            print("actual_tasks:", n_tot_tasks)
            print("required_tasks:", self.settings.data.n_all_tasks)
            return

        shuffled_task_indexes = np.random.permutation(self.settings.data.n_all_tasks)
        # shuffled_tasks = list(range(self.settings.data.n_all_tasks))
        # np.random.shuffle(shuffled_tasks)

        for task_counter, user in enumerate(shuffled_task_indexes):  # enumerate(shuffled_tasks)
            # loading and normalizing the inputs
            zero_indexes = np.where(full_matrix[user, :].toarray() == 0)[1]
            non_zero_indexes = np.nonzero(full_matrix[user, :])[1]
            features = csc_matrix(np.eye(n_jokes))
            # features = np.eye(n_jokes)
            features[zero_indexes, zero_indexes] = 0

            # loading the labels
            labels = full_matrix[user, :].toarray().ravel()


            if task_counter >= self.settings.data.n_tr_tasks:

                # split into training and test
                tr_indexes, ts_indexes = train_test_split(non_zero_indexes, test_size=self.settings.data.ts_points_pct)
                features_tr = features[tr_indexes, :]  # copy.deepcopy(features)  (big and not necessary matrices)
                labels_tr = labels[tr_indexes]  # copy.deepcopy(labels)
                # features_tr[ts_indexes, ts_indexes] = 0
                # labels_tr[ts_indexes] = 0

                features_ts = features[ts_indexes, :]  # copy.deepcopy(features)  (big and not necessary matrices)
                labels_ts = labels[ts_indexes]  # copy.deepcopy(labels)
                # features_ts[tr_indexes, tr_indexes] = 0
                # labels_ts[tr_indexes] = 0

                self.features_tr[user] = features_tr
                self.features_ts[user] = features_ts
                self.labels_tr[user] = labels_tr
                self.labels_ts[user] = labels_ts
            else:
                self.features_tr[user] = features[non_zero_indexes, :]  # copy.deepcopy(features)  (big and not necessary matrices)
                self.labels_tr[user] = labels[non_zero_indexes]  # copy.deepcopy(labels)

        self.tr_task_indexes = shuffled_task_indexes[:self.settings.data.n_tr_tasks]
        self.val_task_indexes = shuffled_task_indexes[
                                self.settings.data.n_tr_tasks:self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks]
        self.test_task_indexes = shuffled_task_indexes[
                                 self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks:self.settings.data.n_all_tasks]
        self.full_matrix = full_matrix

        # self.tr_task_indexes = np.arange(0, self.settings.data.n_tr_tasks)
        # self.val_task_indexes = np.arange(self.settings.data.n_tr_tasks,
        #                                   self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks)
        # self.test_task_indexes = np.arange(self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks,
        #                                    self.settings.data.n_all_tasks)
