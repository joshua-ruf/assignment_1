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
    'base_estimator__criterion': ('entropy',),
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
###########
### SVM ###
###########

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

# -




