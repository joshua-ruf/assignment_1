# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

from helpers import load_data, run_cv, run_oob

# +
X, y, extras = load_data(ebert=True)

# -


FOLDER = 'problem_2/'
RESULTS = []

# +
####################
### RANDOM GUESS ###
####################

from sklearn.dummy import DummyClassifier

print('Always guess most common label:')
DC0 = run_oob(X, y, DummyClassifier(strategy='most_frequent', random_state=0))
RESULTS.append(DC0)
pprint(DC0)

print('\nGuess randomly between True and False, weighted by their distribution in the data:')
DC1 = run_oob(X, y, DummyClassifier(strategy='stratified', random_state=0))
RESULTS.append(DC1)
pprint(DC1)

# -



# +
#####################
### DECISION TREE ###
#####################

from sklearn.tree import DecisionTreeClassifier

parameters = {
    'criterion': ('entropy', 'gini'),
    'splitter': ('best', 'random'),
    'ccp_alpha': np.arange(0, 0.061, 0.01),
    'class_weight': ('balanced',),
    'max_features': ('sqrt', None),
    'min_samples_split': np.arange(2, 12, 1),
    'min_samples_leaf': np.arange(1, 6, 1),
    'max_depth': np.arange(1, 30, 1),
}

DTC = run_cv(X, y, DecisionTreeClassifier(random_state=0), parameters, N=1500, return_train_score=True)

RESULTS.append(DTC[1])
DTC[0].to_csv(f'{FOLDER}DecisionTreeClassifier.csv', index=False)

# -




# +
D = DTC[0].copy()

D.loc[D.param_class_weight != 'balanced', 'param_class_weight'] = 'None'
D.loc[D.param_max_features != 'sqrt', 'param_max_features'] = 'None'

D = D[D.param_class_weight == 'balanced']
D = D[D.param_ccp_alpha <= 0.01]

# """
# - does not seem to make a difference between gini and entropy
# - splitter also doesn't really matter
# - cpp_alpha

# """

param = 'param_' + 'min_samples_leaf'
param1 = 'param_' + 'min_samples_split'
fig, ax = plt.subplots()
# ax = sns.lineplot(
#     D,
#     x=param,
#     y='mean_train_score',
#     label='train',
# )
ax1 = sns.scatterplot(
    D,
    x=param,
    y=param1,
    hue='mean_test_score',
    label='test',
)
# +
params = [
#     "criterion",
#     "splitter",
    "ccp_alpha",
#     "class_weight",
#     "max_features",
    "min_samples_split",
    "min_samples_leaf",
    "max_depth",
]

g = sns.PairGrid(D, vars=[f"param_{x}" for x in params], hue='mean_train_score')
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()
# -



# +
##########################
### BOOSTING: AdaBoost ###
##########################

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

parameters = {
    'base_estimator__criterion': ('entropy', 'gini'),
    'base_estimator__splitter': ('best', 'random'),
    'base_estimator__class_weight': ('balanced',),
    'base_estimator__max_depth': np.arange(1, 7, 1),
    'base_estimator__max_features': ('sqrt', None),
    'base_estimator__ccp_alpha': np.arange(0, 0.051, 0.01),
    'base_estimator__min_samples_split': np.arange(2, 10, 1),
    'base_estimator__min_samples_leaf': np.arange(1, 5, 1),
    'n_estimators': np.append(np.arange(1, 4), np.arange(4, 40, 3)),
}

abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=0), random_state=0)
ABC = run_cv(X, y, abc, parameters, N=1000)

RESULTS.append(ABC[1])
ABC[0].to_csv(f'{FOLDER}AdaBoostClassifier.csv', index=False)



# +
# D = ABC[0].copy()
# D.loc[D.param_base_estimator__class_weight != 'balanced', 'param_base_estimator__class_weight'] = 'None'
# D.loc[D.param_base_estimator__max_features != 'sqrt', 'param_base_estimator__max_features'] = 'None'

# D = D[
#     (D.param_base_estimator__class_weight == 'balanced')
# ]

# """
# balanced class weights best as usual
# splitter seems a mixed bag
# max features also tough to say

# tough to tell beyond that since the four parameters below are all intertwined, probably
# better not to tweak all four at the same time

# """

# # fig, ax = plt.subplots()
# # ax = sns.lineplot(
# #     D,
# #     x='param_n_estimators',
# #     y='mean_test_score',
# #     hue="param_" + 'base_estimator__min_samples_leaf',
# # #     label='train',
# # )

# params = [
#     'base_estimator__max_depth',
#     'base_estimator__ccp_alpha',
#     'base_estimator__min_samples_split',
#     'base_estimator__min_samples_leaf',
# ]

# g = sns.PairGrid(D, vars=[f"param_{x}" for x in params] + ['mean_test_score'])
# g.map_diag(sns.histplot)
# g.map_offdiag(sns.scatterplot)
# g.add_legend()



# +
###########
### SVM ###
###########

from sklearn.svm import SVC

parameters = {
    'C': np.arange(0.1, 3, 0.1),
    'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
    'degree': np.arange(2, 5),  # only for poly
    'gamma': ('scale', 'auto'),  # does not apply to linear
    'coef0': np.arange(0, 2, 0.2),  # applies to poly and sigmoid
    'shrinking': (True, False),
    'class_weight': ('balanced',),
}

SVM = run_cv(X, y, SVC(random_state=0), parameters, N=1000)

RESULTS.append(SVM[1])
SVM[0].to_csv(f'{FOLDER}SVC.csv', index=False)



# + endofcell="--"
# pprint(SVM[1])
# """
# notes:
# - balanced class_weight clearly outperforms None since it f1 precision and recall...
# - rbf clearly outperforms the other kernels -> coef and degree don't apply
# - shrinking is noisy
# - 

# """

# D = SVM[0].copy()
# D.loc[D.param_class_weight != 'balanced', 'param_class_weight'] = 'None'

# D = D[
#     (D.param_class_weight == 'balanced') & 
#     (D.param_kernel == 'rbf')
# ]

# # fig, ax = plt.subplots()
# # ax = sns.lineplot(
# #     D,
# #     x='param_C',
# #     y='mean_test_score',
# #     hue='param_gamma',
# #     label='train',
# # )

# # ax = sns.scatterplot(
# #     D,
# #     x='param_gamma',
# #     y='param_C',
# #     hue='mean_test_score',
# # #     label='train',
# # )

# params = {'shrinking': True, 'gamma': 'scale', 'C': 0.2}

# g = sns.PairGrid(D, vars=[f"param_{x}" for x in params], hue='mean_test_score')
# g.map_diag(sns.histplot)
# g.map_offdiag(sns.scatterplot)
# g.add_legend()
# --



# +
###########
### kNN ###
###########

from sklearn.neighbors import KNeighborsClassifier

parameters = {
    'n_neighbors': np.arange(1, 50),
    'weights': ('uniform',),
}

KNN = run_cv(X, y, KNeighborsClassifier(), parameters, N=500, return_train_score=True)

RESULTS.append(KNN[1])
KNN[0].to_csv(f'{FOLDER}KNN.csv', index=False)


# +
parameters = {
    'n_neighbors': np.arange(5, 50),
    'weights': ('uniform', 'distance'),
}

KNN_oversample = run_cv(X, y, KNeighborsClassifier(), parameters, N=500, oversample=True, return_train_score=True)

KNN_oversample[0]['model'] = KNN_oversample[0]['model'] + " [OVERSAMPLE]"
KNN_oversample[1]['model'] += " [OVERSAMPLE]"

RESULTS.append(KNN_oversample[1])
KNN_oversample[0].to_csv(f'{FOLDER}KNN_oversample.csv', index=False)

# +

# -

R = pd.DataFrame(RESULTS)
R.to_csv(f'{FOLDER}RESULTS.csv', index=False)
print(R)


