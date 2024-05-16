import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn import metrics
import seaborn as sns
import scipy
import math
import platform

# %% Define Key Functions
def compute_prototypicality(data, pp_num, n_clusters=10):

    # switch based on whether 'block_size' exists in data
    if any(data.columns.to_numpy() == 'block_size'):
        measure = '_position'
    else:
        measure = '_colour'

    # extract all values
    vals = np.concatenate([data.loc[:, 'perceived' + measure].to_numpy(), data.loc[:, 'retrieved' + measure].to_numpy()])
    true_vals = np.concatenate([data.loc[:, 'true' + measure].to_numpy(), data.loc[:, 'perceived' + measure].to_numpy()])

    # compute clusters
    sil_score = np.zeros(n_clusters - 1) * np.nan  # predefine loop output
    for k in range(2, n_clusters + 1):
        kmeans = cluster.KMeans(init="random", n_clusters=k, n_init=10, max_iter=300, random_state=42)
        labels = kmeans.fit_predict(vals.reshape(-1, 1))  # fit for given clusters
        sil_score[k - 2] = metrics.silhouette_score(vals.reshape(-1, 1), labels)  # compute score
        del labels, kmeans, k

    # define optimal k as max silhouette score
    optimal_k = np.argmax(sil_score)+2
    del sil_score

    # re-compute clusters with optimal k
    kmeans = cluster.KMeans(init="random", n_clusters=optimal_k, n_init=10, max_iter=300, random_state=42)
    obs_labels = kmeans.fit_predict(vals.reshape(-1, 1))  # fit real data to given clusters
    target_labels = kmeans.predict(true_vals.reshape(-1, 1))  # fit target data for given clusters

    # compute distance to target
    target_distance = []
    for epoch in ['percept', 'retrieved_true']:
        raw_target_dist = abs(data.loc[:, epoch+'_error'].to_numpy())
        target_distance.append(raw_target_dist)
    target_distance = np.concatenate(target_distance, dtype='float')

    # compute distance to prototype
    prototype_distance = np.zeros_like(target_distance)
    for n in range(len(vals)):
        prototype_distance[n] = abs(vals[n] - kmeans.cluster_centers_[target_labels[n]][0])

    # compute distance between target and prototype
    target_prototype_distance = np.zeros_like(target_distance)
    for n in range(len(true_vals)):
        target_prototype_distance[n] = abs(true_vals[n] - kmeans.cluster_centers_[target_labels[n]][0])

    # determine columns in original data to save
    if 'kernel_is_broad' in data.columns:
        columns_to_save = ['trial', 'block', 'confidence', 'kernel_is_broad', 'rt']
    elif 'colorbar' in data.columns:
        columns_to_save = ['trial', 'block', 'confidence', 'colorbar', 'rt']
    elif 'retrieval_order' in data.columns:
        columns_to_save = ['trial', 'block', 'confidence', 'retrieval_order', 'rt']
    else:
        columns_to_save = ['trial', 'block', 'confidence', 'rt']

    # create output data
    df_out = data[columns_to_save].copy()
    df_out = pd.concat([df_out, df_out], axis=0).reset_index(drop=True)  # double up for perception/retrieval distinction
    df_out['pp'] = pp_num
    df_out['optimal_k'] = optimal_k
    df_out['epoch'] = np.repeat(['perception', 'retrieval'], len(data))
    df_out['target_distance'] = target_distance
    df_out['prototype_distance'] = prototype_distance
    df_out['memory'] = np.tile((obs_labels[int(len(obs_labels)/2):] == target_labels[int(len(obs_labels)/2):]).astype('int'), 2)
    df_out['precision'] = np.sqrt((df_out['target_distance'] ** 2) + (df_out['prototype_distance'] ** 2))
    df_out['prototypicality'] = np.array([math.atan2(x, y) for x, y in zip(df_out['target_distance'], df_out['prototype_distance'])]) / (np.pi/2)
    df_out['target_prototype_distance'] = target_prototype_distance

    # scale precision based on distance between target and prototype
    df_out['precision'] = df_out['precision'] / df_out['target_prototype_distance']

    # recode confidence
    if measure == '_colour':
        df_out['confidence'] = (df_out['confidence'] == 'Sure').astype('int')
    elif measure == '_position':
        df_out['confidence'] = (df_out['confidence'] == 'Confident').astype('int')

    # return
    return df_out

def create_query(indep_labels, itvals):
    # create query
    q_idx = []
    for n in np.arange(np.size(indep_labels)):
        if isinstance(itvals[n], str):
            q_idx.append('({}=="{}") & '.format(indep_labels[n], itvals[n]))
        else:
            q_idx.append('({}=={}) & '.format(indep_labels[n], itvals[n]))
    q_idx = ''.join(q_idx)[:-2]  # concatenate
    return q_idx

def scale_vector(x):
    y = (x - min(x)) / (max(x) - min(x))
    return y

# %% Run Participant-Level Analysis
# define datasets
n_datasets = 2

# define root directory
if platform.system() == 'Darwin':
    root_dir = '/Users/ben/Dropbox/work_data/colour_compression/'
else:
    root_dir = 'C:/Users/griffibz/Dropbox/work_data/colour_compression/'

# define loop outputs
group_data = [[] for m in range(n_datasets)]

# cycle through datasets
for dataset in range(n_datasets):

    # get files
    tidy_path = '{}/exp{}/formatted/'.format(root_dir, dataset+1)
    files = os.listdir(tidy_path)
    files = np.sort([f for f in files if f[:2] == 'rt'])

    # cycle through participants
    for pp, f in enumerate(files):

        # read in data
        print('preparing participant {}...'.format(pp+1))
        data = pd.read_csv(tidy_path + f)
        if dataset < 2:
            data = data.query('block > 0').reset_index(drop=True)  # drop training block

        # compute cluster error
        df = compute_prototypicality(data, pp, n_clusters=10)
        group_data[dataset].append(df)
        assert(all(np.isnan(df.rt) == False))

    # concatenate group into single dataframe
    group_data[dataset] = pd.concat(group_data[dataset]).reset_index(drop=True)

# %% Correlate RTs
# predefine correlation dataframe
dfr = []# pd.DataFrame(columns=['dataset', 'pp', 'epoch', 'measure', 'r'])

# cycle through datasets and participants
for dataset in [0, 1]:
    npps = np.unique(group_data[dataset]['pp'])
    for pp in npps:
        for epoch in ['perception', 'retrieval']:
            for measure in ['all', 'sure', 'not_sure']:
                if measure == 'sure':
                    df = group_data[dataset].query('(pp == {}) & (epoch == "{}") & (confidence == 1)'.format(pp, epoch))
                elif measure == 'not_sure':
                    df = group_data[dataset].query('(pp == {}) & (epoch == "{}") & (confidence == 0)'.format(pp, epoch))
                else:
                    df = group_data[dataset].query('(pp == {}) & (epoch == "{}")'.format(pp, epoch))
                x = df['rt'].to_numpy()
                y = df['prototypicality'].to_numpy()
                r, p = scipy.stats.pearsonr(x, y)
                dfr.append(pd.DataFrame(data=np.array([dataset, pp, epoch, measure, r])[:, np.newaxis].transpose(),
                                              columns=['dataset', 'pp', 'epoch', 'measure', 'r']))

# combine
dfr = pd.concat(dfr)

# %% Run Inferential Statistics
# cycle through conditions
fig, ax = plt.subplots(1, 4, sharey='all')
count = 0
dfr['dataset'] = dfr['dataset'].astype('int')
for dataset in [0, 1]:
    for epoch in ['perception', 'retrieval']:
        for measure in ['all', 'sure', 'not_sure']:
            df = dfr.query('(dataset == {}) & (epoch == "{}") & (measure == "{}")'.format(dataset, epoch, measure))
            df['r'] = df['r'].astype('float').to_numpy()
            stat = scipy.stats.ttest_1samp(df['r'], popmean=0)
            print('dataset {}: {} ({}): t({})={:3.2f}, p={:3.3f}'.format(dataset, epoch, measure, stat.df, stat.statistic, stat.pvalue))

            # plot
            if (epoch == 'retrieval') & (measure != 'all'):
                sns.boxplot(df, y='r', ax=ax[count])
                ax[count].set_title('{} {}'.format(dataset, measure))
                ax[count].set_ylim([-0.4, 0.4])
                count += 1

# save plot
plt.show()
fig.savefig('{}/figures/suppfig1.svg'.format(root_dir))

# save data
dfr.to_csv('{}/supplementary_rt_data.csv'.format(root_dir))