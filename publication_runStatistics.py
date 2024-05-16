# %% Import Packages
import pandas as pd
import numpy as np
from statsmodels.stats import anova
import statsmodels.api as sm
import scipy

# %% Prepare Workspace
# define datasets
n_datasets = 6

# define root directory
root_dir = ''

# load data
group_data = pd.read_csv(root_dir + 'group_data.csv')

# %% Get Descriptives
# create descriptives dataframe
descriptives = pd.DataFrame(data=np.zeros([n_datasets*2, 4]), columns=['dataset', 'measure', 'mean', 'sem'])
count = 0

# cycle through datasets
for dataset in range(n_datasets):

    # copy data
    df = group_data.query('(epoch == "perception") & (dataset == {})'.format(dataset)).copy()

    # cycle through measures
    for measure in ['confidence', 'optimal_k']:

        # group by participants
        dat = df.drop(columns=['epoch', 'manipulation']).groupby('pp', as_index=False).mean()[measure].to_numpy()

        # add to dataframe
        descriptives.loc[count, :] = [dataset, measure, np.mean(dat), dat.std() / np.sqrt(len(dat))]
        count = count + 1

# print descriptive
print(descriptives)

# %% Run Main Statistics
# define key variables
predictors = ['confidence', 'epoch']
outcome = 'prototypicality'
repetitions = 'pp'

# cycle through datasets
for dataset in range(n_datasets):

    # get data
    tmp_data = group_data.query('dataset == {}'.format(dataset))

    # group by key variables
    if len(np.unique(tmp_data['manipulation'])) == 1:
        varis = [repetitions] + predictors
        tmp_data = tmp_data.drop(columns='manipulation')
    else:
        varis = [repetitions] + predictors + ['manipulation']
    tmp_data = tmp_data.groupby(varis, as_index=False).mean()

    # drop participants with missing data
    for pp in np.unique(tmp_data['pp']):
        if sum(tmp_data['pp'] == pp) != 2**(len(varis)-1):
            tmp_data = tmp_data[tmp_data['pp'] != pp]
            print('dropping sub-{} in dataset {}'.format(pp, dataset))
        del pp

    # export to csv
    df = tmp_data.reset_index(drop=True).drop(columns=['Unnamed: 0', 'trial', 'block', 'dataset', 'optimal_k', 'target_distance', 'prototype_distance', 'memory', 'scaled_pos'])
    df['confidence'] = df['confidence'].astype('str')
    if 'manipulation' in df:
        df = df.pivot(index='pp', columns=['confidence', 'epoch', 'manipulation'], values='prototypicality')
    else:
        df = df.pivot(index='pp', columns=['confidence', 'epoch'], values='prototypicality')
    df.columns = list(map("_".join, df.columns))
    df.to_csv('{}/stats_exp{}.csv'.format(root_dir, dataset+1))

    # run anova
    res = anova.AnovaRM(data=tmp_data.reset_index(drop=True),
                        depvar=outcome,
                        within=varis[1:],
                        subject=repetitions).fit()

    # print summary
    print(res.summary())

    # if experiment 6, check that retrieval confidence effect is present
    if dataset == 5:
        print('\n\n--- Check Retrieval Effect in Exp. 6 ---')
        for epoch in ['perception', 'retrieval']:
            x = tmp_data.query('(confidence == 1) & (epoch == "{}")'.format(epoch)).prototypicality.to_numpy()
            y = tmp_data.query('(confidence == 0) & (epoch == "{}")'.format(epoch)).prototypicality.to_numpy()
            stats = scipy.stats.ttest_rel(x, y)
            print('{} for Exp. 6: "Sure" > "Not Sure": t({})={}, p={:03.3f}'.format(epoch, stats.df, stats.statistic, stats.pvalue))

    # tidy
    del res, tmp_data, varis, dataset

# tidy
del predictors,outcome, repetitions

# %% Correlate Perceptual/Retrieval Prototypicality
# predefine correlation dataframe
dfr = pd.DataFrame(columns=['dataset', 'pp', 'r'])

# cycle through datasets
for dataset in range(n_datasets):

    # get current data, and identify number of participants
    tmp_data = group_data.query('dataset == {}'.format(dataset))
    npps = np.unique(tmp_data['pp'])

    # cycle through participants and confidence types
    for pp in npps:

        # restrict data to participant
        df = tmp_data.query('(pp == {})'.format(pp))

        # correlate prototypicality between perception (x) and prototypicality (y)
        x = df.query('epoch=="perception"')['prototypicality'].to_numpy()
        y = df.query('epoch=="retrieval"')['prototypicality'].to_numpy()
        if (len(x) < 2) | (len(y) < 2):
            r = np.nan
        else:
            r, _ = scipy.stats.pearsonr(x, y)
            del _

        # update dataframe
        dfr = pd.concat([dfr, pd.DataFrame(data=np.array([dataset, pp, r])[:, np.newaxis].transpose(),
                                           columns=['dataset', 'pp', 'r'])], ignore_index=True)

        # tidy
        del r, x, y, pp, df
    del tmp_data, npps, dataset

# prepare for statistical test
dfr['dataset'] = dfr['dataset'].astype('int')
print('\n\n--------- Correlating Perceptual and Retrieved Prototypicality ---------')

# cycle through datasets and conditions
for dataset in range(n_datasets):

    # get relevant data
    df = dfr.query('dataset == {}'.format(dataset))

    # export to csv
    df.to_csv('{}/encret_correlation_exp{}.csv'.format(root_dir, dataset+1))

    # run and report test
    stat = scipy.stats.ttest_1samp(df['r'], popmean=0, nan_policy='omit')
    print('dataset {}: t({})={:3.2f}, p={:3.3f}'.format(dataset, stat.df, stat.statistic, stat.pvalue))

    # tidy
    del df, stat
del dfr

# %% Report Confidence for Exp. 4
# report summary statistics
print('\n\n--------- Summary of Confidence (Exp. 4) ---------')
print('Broad Kernels: mean = {:3.3f} (std: {:3.3f})'.format(group_data.query('manipulation=="kernel_broad"').groupby('pp').mean(numeric_only=True)['confidence'].mean(),
                                                            group_data.query('manipulation=="kernel_broad"').groupby('pp').mean(numeric_only=True)['confidence'].std()))
print('Narrow Kernels: mean = {:3.3f} (std: {:3.3f})'.format(group_data.query('manipulation=="kernel_narrow"').groupby('pp').mean(numeric_only=True)['confidence'].mean(),
                                                             group_data.query('manipulation=="kernel_narrow"').groupby('pp').mean(numeric_only=True)['confidence'].std()))

# %% Run Impact of Controls on Confidence
# cycle through datasets
print('\n\n--------- Impact of Manipulations on Confidence ---------')
for dataset in range(n_datasets):

    # grab copy of dataset
    tmp_data = group_data.query('dataset == {}'.format(dataset))

    # skip if no manipulation
    if len(np.unique(tmp_data['manipulation'])) == 1:
        del tmp_data, dataset
        continue

    # get grouped data
    tmp_data = tmp_data.query('epoch == "retrieval"')
    tmp_data = tmp_data.drop(columns='epoch').groupby(['pp', 'manipulation'], as_index=False).mean()

    # export to csv
    df = tmp_data.reset_index(drop=True).drop(columns=['Unnamed: 0', 'trial', 'block', 'dataset', 'optimal_k', 'target_distance', 'prototype_distance', 'memory', 'scaled_pos', 'prototypicality'])
    df = df.pivot(index='pp', columns=['manipulation'], values='confidence')
    df.to_csv('{}/confidence_manip_exp{}.csv'.format(root_dir, dataset + 1))

    # run paired t-test
    vals = np.sort(np.unique(tmp_data['manipulation']))
    a = tmp_data['confidence'][tmp_data['manipulation'] == vals[0]].values
    b = tmp_data['confidence'][tmp_data['manipulation'] == vals[1]].values
    t, p = scipy.stats.ttest_rel(a, b)

    # print summary
    print('Exp. {} ({} > {}): t({:d}) = {:3.3f}, p = {:3.3f}'.format(dataset, vals[0], vals[1], len(a)-1, t, p))

    # tidy
    del dataset, vals, a, b, t, p, tmp_data

# tidy
del n_datasets

# %% Test if Prospective Confidence persists after controlling for perceptual distortion
# loop through participants
dat = group_data.query('(dataset == 5)')
beta = []
for pp in np.unique(dat['pp']):

    # get key variables
    percept_bias = dat.query('(pp == {}) & (epoch == "perception")'.format(pp))['prototypicality'].to_numpy()
    retrieval_bias = dat.query('(pp == {}) & (epoch == "retrieval")'.format(pp))['prototypicality'].to_numpy()
    confidence = dat.query('(pp == {}) & (epoch == "retrieval")'.format(pp))['confidence'].to_numpy()

    # create design matrix
    X = np.ones([len(confidence), 3])
    X[:, 1] = percept_bias
    X[:, 2] = confidence
    y = retrieval_bias

    # fit model (if variability in data)
    if np.all(X[:, 0] == X[:, 2]):
        continue
    res = sm.OLS(y, X).fit()
    beta.append(res.params)

# run t-tests
print('\n\n\n------ Controlling for Perceptual Distortion ------')
beta = np.array(beta)
stat = scipy.stats.ttest_1samp(beta[:, 1], popmean=0, nan_policy='omit')
print('perceptual bias: t({})={:3.2f}, p={:3.3f}'.format(stat.df, stat.statistic, stat.pvalue))
stat = scipy.stats.ttest_1samp(beta[:, 2], popmean=0, nan_policy='omit')
print('confidence: t({})={:3.2f}, p={:3.3f}'.format(stat.df, stat.statistic, stat.pvalue))

# update user
print('\n\ncomplete...')
