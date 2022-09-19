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
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

FILE = '../20220918_data.csv'


# -

def load_data(file=FILE):
    df = pd.read_csv(file)
    df.drop(columns=['hire_date'], inplace=True)
    for var in ('gender', 'ethnicity'):
        temp = pd.get_dummies(df[var], prefix=var, drop_first=True)
        df.drop(columns=[var], inplace=True)
        df = df.join(temp)
        
    y = df['terminated_in_first_6_months']
    X = df.drop(columns=['terminated_in_first_6_months'])
    
    return X, y


def apply_pca(data, cols, prop_var_explained=0.75):
    
    D = data.copy()
    
    temp = D.iloc[:, cols].values
    temp = MinMaxScaler().fit_transform(temp)

    pca = PCA(n_components=temp.shape[1])
    pca.fit(temp)

    var_explained = pca.explained_variance_ratio_.cumsum()

    components = np.argmax(var_explained >= prop_var_explained)

    temp2 = PCA(components).fit_transform(temp)
    temp2 = pd.DataFrame(temp2)
    temp2.columns = [f'pcomponent_{i}' for i in temp2.columns]
    
    return D.iloc[:, ~cols].join(temp2)


