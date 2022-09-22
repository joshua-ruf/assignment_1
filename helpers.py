# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

FILE = '../20220918_data.csv'


# -

def load_data(file=FILE):
    df = pd.read_csv(file)
    df.drop(columns=['hire_date'], inplace=True)
    
    # create dummy variables out of the categorical demographic features
    for var in ('gender', 'ethnicity'):
        temp = pd.get_dummies(df[var], prefix=var, drop_first=True)
        df.drop(columns=[var], inplace=True)
        df = df.join(temp)
        
    y = df['terminated_in_first_6_months']
    X = df.drop(columns=['terminated_in_first_6_months'])
    
    # scale all inputs between -1 and 1
    X = MinMaxScaler((-1, 1)).fit_transform(X)
    
    return X, y


def run_cv(X, y, base_estimator, parameters, N=100, variance_threshold=0, **kwargs):
    """
    X are features
    y is labels
    base_estimator is the model with which to run CV
    parameters is a dictionary of parameters to pass to skelarn's CV function
    N is the number of iterations
    variance_threshold is the minimum variance of a feature to be included as a feature. By default
    it just removes zero variance features
    """
#     X = VarianceThreshold(variance_threshold).fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)
    
    # run cross validation, sampling N choices over the parameter set
    cv = RandomizedSearchCV(
        base_estimator,
        parameters,
        n_iter=N, scoring='f1', n_jobs=-1,
        random_state=0, verbose=1, **kwargs,
    )
    cv.fit(X_train, y_train)
    
    # print the best params and their corresponding average f1 score across folds
    print({
        'best_params': cv.best_params_,
        'best_score': cv.best_score_,
    })
    
    # store all results for exploration
    frame = pd.DataFrame(cv.cv_results_)
    frame['model'] = str(base_estimator)
    
    # get the best model and run over the entire train and test set for comparison
    best_estimator = cv.best_estimator_
    score = {
        'model': str(base_estimator),
        'full_train_f1': f1_score(best_estimator.predict(X_train), y_train),
        'full_test_f1': f1_score(best_estimator.predict(X_test), y_test),
        'full_train_accuracy': best_estimator.score(X_train, y_train),
        'full_test_accuracy': best_estimator.score(X_test, y_test),
    }
    
    return (frame, score, best_estimator)


pd.read_csv(FILE).columns



