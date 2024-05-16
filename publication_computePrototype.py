# import packages
import pandas as pd
import numpy as np
import os
from sklearn import cluster
from sklearn import metrics
import math
import platform

# %% Run Participant-Level Analysis
# define key variables
n_datasets = 6
n_clusters = 10  # set max. clusters

# define root directory
root_dir = ''

# define loop outputs
group_data = [[] for m in range(n_datasets)]

# cycle through datasets
for dataset in range(n_datasets):

    # get files
    tidy_path = '{}/exp{}/formatted/'.format(root_dir, dataset+1)
    files = os.listdir(tidy_path)
    files = np.sort([f for f in files if f[:3] == 'sub'])

    # cycle through participants
    for pp, f in enumerate(files):

        # read in data
        print('preparing participant {} of dataset {}...'.format(pp+1, dataset+1))
        data = pd.read_csv(tidy_path + f)
        if (dataset == 0) | (dataset == 2):
            data = data.query('block > 0').reset_index(drop=True)  # drop training block

        # switch based on whether 'block_size' exists in data
        if dataset == 2:
            measure = '_position'
        else:
            measure = '_colour'

        # extract responses (perceived, retrieved) and references (true)
        vals = np.concatenate([data.loc[:, 'perceived' + measure].to_numpy(), data.loc[:, 'retrieved' + measure].to_numpy()])
        true_vals = np.concatenate([data.loc[:, 'true' + measure].to_numpy(), data.loc[:, 'perceived' + measure].to_numpy()])
        del measure

        # determine distance to label in leave-one-out fashion
        target_distance = np.zeros_like(vals)
        prototype_distance = np.zeros_like(vals)
        scaled_pos = np.zeros_like(vals)
        optimal_k = np.zeros_like(vals).astype('int')
        memory = np.zeros_like(vals).astype('int')
        for trl in range(len(vals)):

            # split data
            train_vals = vals[np.arange(len(vals)) != trl]
            test_vals = vals[np.arange(len(vals)) == trl][0]
            test_true = true_vals[np.arange(len(vals)) == trl][0]

            # compute clusters for k ranging from 2 to [n_clusters]
            sil_score = np.zeros(n_clusters - 1) * np.nan  # predefine loop output
            for k in range(2, n_clusters + 1):
                kmeans = cluster.KMeans(init="random", n_clusters=k, n_init=10, max_iter=300, random_state=42)
                labels = kmeans.fit_predict(train_vals.reshape(-1, 1))  # fit for given clusters
                sil_score[k - 2] = metrics.silhouette_score(train_vals.reshape(-1, 1), labels)  # compute score
                del labels, kmeans, k

            # define optimal k as max silhouette score
            optimal_k[trl] = np.argmax(sil_score) + 2  # index of zero == optimal k of 2
            del sil_score

            # re-compute clusters with optimal k
            kmeans = cluster.KMeans(init="random", n_clusters=optimal_k[trl], n_init=10, max_iter=300, random_state=42)
            kmeans.fit(train_vals.reshape(-1, 1))  # fit real data to given clusters
            obs_label = kmeans.predict(test_vals.reshape(-1, 1))[0]
            target_label = kmeans.predict(test_true.reshape(-1, 1))[0]  # fit target data for given clusters

            # compute distance to target and to prototype
            target_distance[trl] = abs(test_vals - test_true)
            prototype_distance[trl] = abs(test_vals - kmeans.cluster_centers_[obs_label][0])
            memory[trl] = int(obs_label == target_label)

            # get relative position
            point0 = kmeans.cluster_centers_[obs_label][0]
            point1 = test_true
            point_sign = np.sign(int(point0 < point1) - 0.5)
            point0 = point0 * point_sign
            point1 = point1 * point_sign
            x = test_vals * point_sign
            x = (x - point0) / (point1 - point0)
            scaled_pos[trl] = x
            del point0, point1, point_sign, x

            # tidy
            del train_vals, test_vals, test_true, kmeans, obs_label, target_label, trl

        # tidy
        del vals, true_vals

        # create output data
        df = data[['trial', 'block', 'confidence']].copy()
        df = pd.concat([df, df], axis=0).reset_index(drop=True)  # double up for perception/retrieval distinction
        df['dataset'] = dataset
        df['pp'] = pp
        df['optimal_k'] = optimal_k
        df['epoch'] = np.repeat(['perception', 'retrieval'], len(data))
        df['target_distance'] = target_distance
        df['prototype_distance'] = prototype_distance
        df['memory'] = memory
        df['prototypicality'] = np.array([math.atan2(x, y) for x, y in zip(target_distance, prototype_distance)]) / (np.pi / 2)
        df['scaled_pos'] = scaled_pos
        del optimal_k, target_distance, prototype_distance, scaled_pos

        # add in manipulation column
        for trl in range(len(data)):
            if dataset == 1:
                if data.loc[trl, 'colorbar'] == 1:
                    val = 'colorbar_forward'
                else:
                    val = 'colorbar_backward'
            elif dataset == 3:
                if data.loc[trl, 'retrieval_order'] == 1:
                    val = 'order_retrieval'
                else:
                    val = 'order_confidence'
            elif dataset == 4:
                if data.loc[trl, 'kernel_is_broad'] == 1:
                    val = 'kernel_broad'
                else:
                    val = 'kernel_narrow'
            else:
                val = 'none'
            df.loc[trl, 'manipulation'] = val
            df.loc[trl+len(data), 'manipulation'] = val
            del val, trl

        # recode confidence
        if dataset != 2:
            df['confidence'] = (df['confidence'] == 'Sure').astype('int')
        elif dataset == 2:
            df['confidence'] = (df['confidence'] == 'Confident').astype('int')

        # compute cluster error
        group_data[dataset].append(df)

        # tidy up
        del df, data, pp, f

    # concatenate group into single dataframe
    group_data[dataset] = pd.concat(group_data[dataset]).reset_index(drop=True)
    del dataset, tidy_path, files

# combine datasets
group_data = pd.concat(group_data).reset_index(drop=True)

# save data
group_data.to_csv(root_dir + 'group_data.csv')

# tidy up
del group_data, root_dir, n_clusters, n_datasets

# update user
print('complete...')