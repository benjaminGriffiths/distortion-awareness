import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn import metrics
import seaborn as sns
from statsmodels.stats import anova
from shapely.geometry import Polygon
import itertools
import platform
import scipy

# %% Define Key Functions
def compute_cluster_error(data, pp_num, n_clusters=10):

    # switch based on whether 'block_size' exists in data
    cluster_sizes = np.arange(n_clusters)+1
    if any(data.columns.to_numpy() == 'block_size'):
        measure = '_position'
        variables = ['pp', 'k', 'condition', 'confidence', 'block_size', 'value']
        sub_conditions = [[pp_num], cluster_sizes, ['true', 'perceived', 'retrieved'],
                          ['All', 'Confident', 'Not Confident'], [6, 8, 10], [np.nan]]
    else:
        measure = '_colour'
        variables = ['pp', 'k', 'condition', 'confidence', 'value']
        sub_conditions = [[pp_num], cluster_sizes, ['true', 'perceived', 'retrieved'],
                          ['All', 'Sure', 'Unsure', 'Guess'], [np.nan]]

    # define participant dataframe
    var_iterations = list(itertools.product(*sub_conditions))
    df = pd.DataFrame(data=np.zeros([np.size(var_iterations, axis=0), np.size(variables)]), columns=variables)
    df_sil = df.copy()

    # iterate through possible combinations of conditions
    for i, itervals in enumerate(var_iterations):

        # add iteration data to dataframe
        df.loc[i, :] = itervals
        df_sil.loc[i, :] = itervals

        # extract query index in raw data
        if measure == '_colour':
            q_idx = 'confidence=="{}"'.format(itervals[3])
        else:
            q_idx = '(confidence=="{}") & (block_size=={})'.format(itervals[3], itervals[4])

        # extract responses
        if itervals[3] == 'All':
            vals = data[itervals[2] + measure].to_numpy()
        else:
            vals = data.query(q_idx)[itervals[2] + measure].to_numpy()

        # if there are more than [n_cluster] responses
        if np.size(vals) > n_clusters:

            # compute cluster error
            kmeans = cluster.KMeans(init="random", n_clusters=itervals[1], n_init=10, max_iter=300, random_state=42)
            kmeans.fit(vals.reshape(-1, 1))
            assigned_cluster_centre = [kmeans.cluster_centers_[x][0] for x in kmeans.labels_]
            distance_to_cluster = vals - assigned_cluster_centre
            clus_error = abs(distance_to_cluster).mean()

            # add to dataframne
            df.loc[i, 'value'] = clus_error

            # compute silhouette score
            if itervals[1] != 1:
                df_sil.loc[i, 'value'] = metrics.silhouette_score(vals.reshape(-1, 1), kmeans.labels_)

            # tidy up
            del kmeans, assigned_cluster_centre, distance_to_cluster, clus_error

        else:
            df.loc[i, 'value'] = np.nan

        # tidy up
        del vals, i, itervals

    # determine peak silhouette
    if any(data.columns.to_numpy() == 'block_size'):
        variables = ['pp', 'condition', 'confidence', 'block_size', 'value']
        sil_conditions = [[pp_num], ['true', 'perceived', 'retrieved'], ['All', 'Confident', 'Not Confident'],
                          [6, 8, 10], [np.nan]]
    else:
        variables = ['pp', 'condition', 'confidence', 'value']
        sil_conditions = [[pp_num], ['true', 'perceived', 'retrieved'], ['All', 'Sure', 'Unsure', 'Guess'], [np.nan]]

    # define participant dataframe
    var_iterations = list(itertools.product(*sil_conditions))
    df_peak = pd.DataFrame(data=np.zeros([np.size(var_iterations, axis=0), np.size(variables)]), columns=variables)

    # iterate through possible combinations of conditions
    for i, itervals in enumerate(var_iterations):

        # add iteration data to dataframe
        df_peak.loc[i, :] = itervals

        # get data for subcondition
        query = create_query(variables[:-1], itervals[:-1])
        vals = df_sil.query(query).value[1:].to_numpy()
        k = df_sil.query(query).k[1:].to_numpy()

        # determine peak
        if not all(np.isnan(vals)):
            max_val = np.argmax(vals)
            df_peak.loc[i, 'value'] = k[max_val]

    # tidy up and return
    return df, df_peak

def compute_auc(k_error, pp_num):

    # define condition labels
    variables = list(k_error.columns)
    indep_vars = [x for x in variables if (x != 'value') & (x != 'k')]
    sub_conditions = [np.unique(k_error[x]).tolist() for x in indep_vars] + [[0]]

    # define participant dataframe
    var_iterations = list(itertools.product(*sub_conditions))
    df = pd.DataFrame(data=np.zeros((np.size(var_iterations, axis=0), np.size(indep_vars)+1)),
                      columns=indep_vars + ['value'])

    # iterate through possible combinations of conditions
    for i, itervals in enumerate(var_iterations):

        # create query
        q_idx = create_query(indep_vars, itervals)

        # extract values for specified participant/condition/confidence
        k_vals = k_error.query(q_idx)['value'].to_numpy()

        # compute auc (if not NaN)
        if any([pd.isna(val) for val in k_vals]):
            k_auc = np.nan
        else:
            a = k_vals
            a0 = np.zeros_like(a)
            x = np.arange(0, np.size(a))
            x0 = np.flip(x)
            k_auc = Polygon(zip(np.append(a, a0), np.append(x, x0))).area
            del a, a0, x, x0

        # add to group
        df.loc[i, :] = itervals
        df.loc[i, 'value'] = k_auc

    # reference to 'all'
    for i, itervals in enumerate(var_iterations):

        # get reference iterations
        ref_itervals = list(itervals)
        cond_label_idx = np.where(np.array(indep_vars) == 'condition')[0][0]
        conf_label_idx = np.where(np.array(indep_vars) == 'confidence')[0][0]
        if ref_itervals[cond_label_idx] == 'true':
            continue
        else:
            ref_itervals[cond_label_idx] = 'true'
            ref_itervals[conf_label_idx] = 'All'

        # create query
        q_idx = create_query(indep_vars, itervals)
        ref_idx = create_query(indep_vars, ref_itervals)

        # get relevant indices
        obs_val = df.query(q_idx)['value'].to_numpy()
        ref_val = df.query(ref_idx)['value'].to_numpy()

        # get difference
        df.loc[i, 'value'] = ((obs_val - ref_val) / ref_val) * -1

    # drop 'all' condition
    df = df[df['condition'] != 'true'].reset_index(drop=True)
    return df

#
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


# %% Run Participant-Level Analysis
# define datasets
n_datasets = 1

# define loop outputs
group_error = []
group_silhouette = []
group_auc = []

# get files
if platform.system() == 'Darwin':
    tidy_path = '/Users/ben/Dropbox/work_data/colour_compression/exp1/formatted/'
else:
    tidy_path = 'C:/Users/griffibz/Dropbox/work_data/colour_compression/exp1/formatted/'
files = os.listdir(tidy_path)
files = np.sort([f for f in files if f[:3] == 'sub'])

# cycle through participants
for pp, f in enumerate(files):

    # read in data
    print('preparing participant {}...'.format(pp+1))
    data = pd.read_csv(tidy_path + f)
    data = data.query('block > 0').reset_index(drop=True)  # drop training block

    # compute cluster error
    print('computing cluster error')
    df_error, sil_score = compute_cluster_error(data, pp, n_clusters=10)
    group_error.append(df_error)  # append to group
    group_silhouette.append(sil_score)

    # input variables
    print('computing area under the curve')
    df_auc = compute_auc(df_error, pp)
    group_auc.append(df_auc)  # append to group

    # update user
    print('participant {} for {} complete...\n'.format(pp+1, np.size(files)))
    del pp, f, df_error, sil_score, df_auc, data

# concatenate group into single dataframe
group_error = pd.concat(group_error).reset_index(drop=True)
group_silhouette = pd.concat(group_silhouette).reset_index(drop=True)
group_auc = pd.concat(group_auc).reset_index(drop=True)

# %% Run Group-Level Area-Under-Curve Analysis
# cycle through datasets (excluding demo)
# input variables
df = group_auc
subjects = 'pp'
dep_var = 'value'
indep_vars = ['condition', 'confidence']

# drop 'All' condition
df = df[df['confidence'] != 'All']

# average over conditions not included in variable list
var_list = [subjects] + indep_vars
spare_vars = [x for x in df.columns if all(x != np.array(var_list))]
df = df.groupby(var_list, as_index=False).mean()
df = df.drop(columns=[x for x in spare_vars if x != dep_var])

# conduct anova
good_pps = np.where([any(df[df['pp'] == x]['value'].isna()) == 0 for x in np.unique(df['pp'])])[0]
df_anova = pd.concat([df[df['pp'] == x] for x in good_pps])
results = anova.AnovaRM(data=df_anova, depvar=dep_var, subject=subjects, within=indep_vars).fit()
print(results.summary())

# do pairwise t-tests
import scipy
x = df_anova.query('(confidence == "Sure") & (condition == "retrieved")').value.to_numpy()
y = df_anova.query('(confidence == "Sure") & (condition == "perceived")').value.to_numpy()
stats = scipy.stats.ttest_rel(x, y)

# export to csv
df = df_anova.reset_index(drop=True)
df = df.pivot(index='pp', columns=['confidence', 'condition'], values='value')
df.columns = list(map("_".join, df.columns))
df.to_csv('C:/Users/griffibz/Dropbox/work_data/colour_compression/prereg_stats.csv')

# %% Plot
# plot
fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey='all')
sns.swarmplot(data=group_auc[group_auc['confidence'] != 'All'], x='condition', y='value', hue='confidence',
              dodge=True, hue_order=['Sure', 'Unsure', 'Guess'], ax=ax[0])
sns.boxplot(data=group_auc[group_auc['confidence'] != 'All'], x='condition', y='value', hue='confidence',
              hue_order=['Sure', 'Unsure', 'Guess'], ax=ax[1], whis=np.inf)
plt.title('Experiment 1')
plt.ylim([0, 0.75])
plt.yticks([0, 0.25, 0.5, 0.75])
plt.tight_layout()
fig.savefig('tmp.svg')


del fig, ax
