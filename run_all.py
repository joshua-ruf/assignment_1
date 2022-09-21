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

from helpers import load_data, run_cv
# -

X, y = load_data()


# +
#####################
### DECISION TREE ###
#####################

from sklearn.tree import DecisionTreeClassifier

parameters = {
    'criterion': ('entropy', 'gini'),
    'splitter': ('best', 'random'),
    'ccp_alpha': np.arange(0, 0.061, 0.01),
    'class_weight': ('balanced', None),
    'max_features': ('sqrt', None),
    'min_samples_split': np.arange(2, 12, 1),
    'min_samples_leaf': np.arange(1, 6, 1),
    'max_depth': np.arange(1, 30, 1),
}

DTC = run_cv(X, y, DecisionTreeClassifier(random_state=0), parameters, N=1000)


# +
##########################
### BOOSTING: AdaBoost ###
##########################

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

parameters = {
    'base_estimator__criterion': ('entropy', 'gini'),
    'base_estimator__splitter': ('best', 'random'),
    'base_estimator__class_weight': ('balanced', None),
    'base_estimator__max_depth': np.arange(1, 7, 1),
    'base_estimator__max_features': ('sqrt', None),
    'base_estimator__ccp_alpha': np.arange(0, 0.051, 0.01),
    'base_estimator__min_samples_split': np.arange(2, 10, 1),
    'base_estimator__min_samples_leaf': np.arange(1, 5, 1),
    'n_estimators': np.append(np.arange(1, 4), np.arange(4, 40, 3)),
}

abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=0), random_state=0)
ABC = run_cv(X, y, abc, parameters, N=1000)


# +
D = ABC[0].copy()
D.loc[D.param_base_estimator__class_weight != 'balanced', 'param_base_estimator__class_weight'] = 'None'
D.loc[D.param_base_estimator__max_features != 'sqrt', 'param_base_estimator__max_features'] = 'None'

D = D[
    (D.param_base_estimator__class_weight == 'balanced')
]

"""
balanced class weights best as usual
splitter seems a mixed bag
max features also tough to say

tough to tell beyond that since the four parameters below are all intertwined, probably
better not to tweak all four at the same time

"""

# fig, ax = plt.subplots()
# ax = sns.lineplot(
#     D,
#     x='param_n_estimators',
#     y='mean_test_score',
#     hue="param_" + 'base_estimator__min_samples_leaf',
# #     label='train',
# )

params = [
    'base_estimator__max_depth',
    'base_estimator__ccp_alpha',
    'base_estimator__min_samples_split',
    'base_estimator__min_samples_leaf',
]

g = sns.PairGrid(D, vars=[f"param_{x}" for x in params] + ['mean_test_score'])
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()



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
    'class_weight': ('balanced', None),
}

SVM = run_cv(X, y, SVC(random_state=0), parameters, N=1000)


# +
pprint(SVM[1])
"""
notes:
- balanced class_weight clearly outperforms None since it f1 precision and recall...
- rbf clearly outperforms the other kernels -> coef and degree don't apply
- shrinking is noisy
- 

"""

D = SVM[0].copy()
D.loc[D.param_class_weight != 'balanced', 'param_class_weight'] = 'None'

D = D[
    (D.param_class_weight == 'balanced') & 
    (D.param_kernel == 'rbf')
]

# fig, ax = plt.subplots()
# ax = sns.lineplot(
#     D,
#     x='param_C',
#     y='mean_test_score',
#     hue='param_gamma',
#     label='train',
# )

# ax = sns.scatterplot(
#     D,
#     x='param_gamma',
#     y='param_C',
#     hue='mean_test_score',
# #     label='train',
# )

params = {'shrinking': True, 'gamma': 'scale', 'C': 0.2}

g = sns.PairGrid(D, vars=[f"param_{x}" for x in params], hue='mean_test_score')
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()


# -



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

# -

pprint(KNN[1])
"""
- training f1 and accuracy are both 1.0 so this thing is very overfit when weights is distance
- why would this be the case?
"""
fig, ax = plt.subplots()
ax = sns.lineplot(
    KNN[0],
    x='param_n_neighbors',
    y='mean_train_score',
    label='train',
)
ax1 = sns.lineplot(
    KNN[0],
    x='param_n_neighbors',
    y='mean_test_score',
    label='test',
)




