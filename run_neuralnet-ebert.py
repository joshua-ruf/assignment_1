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
from pprint import pprint
from collections import Counter

from helpers import load_data

FOLDER = 'problem_2/'

# +
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterSampler

X, y, extra = load_data(ebert=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)


# +
######################
### NEURAL NETWORK ###
######################

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset


# +
X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
y_train, y_test = torch.Tensor(y_train.astype(float).values), torch.Tensor(y_test.astype(float).values)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# +
c = Counter(int(x) for x in y_train)

samples_weight = 1 / torch.tensor([c[int(t)] for t in y_train])

sampler = WeightedRandomSampler(samples_weight, len(samples_weight))


# +
######### CV #############

INPUT_DIM = X.shape[1]
OUTPUT_DIM = 1
VERBOSE = False

parameter_grid = {
    'batch_size': 2 ** np.arange(2, 6),
    'hidden_dim': 2 ** np.arange(2, 6),
    'learning_rate': np.arange(0.005, 0.025, 0.005),
    'dropout_rate': np.arange(0.0, 0.41, 0.05),
    'momentum': np.arange(0.0, 1.01, 0.1),
    'epochs': np.arange(50, 501, 50),
}

class NNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(NNet, self).__init__()
        self.cf1 = torch.nn.Linear(input_dim, hidden_dim)
        self.cf2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cf3 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.sigmoid(self.cf1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.cf2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.cf3(x))
        return x

criterion = torch.nn.BCELoss()

RESULTS = []
N = 200
for j, parameters in enumerate(ParameterSampler(parameter_grid, n_iter=N, random_state=0)):
    if j % 10 == 0:
        print(f"{j/N:.3f}")
    
    train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=int(parameters['batch_size']),
                                  sampler=sampler)
    test_dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=int(parameters['batch_size']))

    model = NNet(
        INPUT_DIM,
        parameters['hidden_dim'],
        OUTPUT_DIM,
        parameters['dropout_rate'],
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=parameters['learning_rate'],
        momentum=parameters['momentum'],
    )
    
    # to determine how many epochs are needed
    epoch_number = 0
    best_vloss = np.inf
    best_dict = None

    for epoch in range(parameters['epochs']):

        # training
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass, backward pass, optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        running_loss /= i + 1

        # testing
        model.eval()
        running_vloss = 0.0
        for i, vdata in enumerate(test_dataloader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss

        running_vloss /= i + 1

        if VERBOSE:
            print(f"[{epoch + 1}] train loss: {running_loss:.3f}, test loss: {running_vloss:.3f}")

        if running_vloss < best_vloss:
            best_vloss = running_vloss
            epoch_number = epoch
            best_dict = model.state_dict()
            
    model.load_state_dict(best_dict)
    model.eval()
    
    RESULTS.append({
        **parameters,
        'best_vloss': float(best_vloss),
        'best_epoch': epoch_number,
        'training_f1_score': f1_score(model(X_train).round().detach().numpy(), y_train),
        'testing_f1_score': f1_score(model(X_test).round().detach().numpy(), y_test)
    })
    
# -


R = pd.DataFrame(RESULTS)
R.to_csv(f'{FOLDER}NNet.csv', index=False)

# +
best_parameters = R.sort_values('best_vloss').iloc[0, :-4].to_dict()
VERBOSE=True
######### CV #############

INPUT_DIM = X.shape[1]
OUTPUT_DIM = 1
VERBOSE = False

parameter_grid = {
    'batch_size': 2 ** np.arange(2, 6),
    'hidden_dim': 2 ** np.arange(2, 6),
    'learning_rate': np.arange(0.005, 0.025, 0.005),
    'dropout_rate': np.arange(0.0, 0.41, 0.05),
    'momentum': np.arange(0.0, 1.01, 0.1),
    'epochs': np.arange(50, 501, 50),
}

class NNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(NNet, self).__init__()
        self.cf1 = torch.nn.Linear(input_dim, hidden_dim)
        self.cf2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cf3 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.sigmoid(self.cf1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.cf2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.cf3(x))
        return x

criterion = torch.nn.BCELoss()

RESULTS = []
N = 200
for j, parameters in enumerate(ParameterSampler(parameter_grid, n_iter=N, random_state=0)):
    if parameters != best_parameters:
        continue
    
    if j % 10 == 0:
        print(f"{j/N:.3f}")
    
    train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=int(parameters['batch_size']),
                                  sampler=sampler)
    test_dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=int(parameters['batch_size']))

    model = NNet(
        INPUT_DIM,
        parameters['hidden_dim'],
        OUTPUT_DIM,
        parameters['dropout_rate'],
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=parameters['learning_rate'],
        momentum=parameters['momentum'],
    )
    
    # to determine how many epochs are needed
    epoch_number = 0
    best_vloss = np.inf
    best_dict = None

    for epoch in range(parameters['epochs']):

        # training
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass, backward pass, optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        running_loss /= i + 1

        # testing
        model.eval()
        running_vloss = 0.0
        for i, vdata in enumerate(test_dataloader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss

        running_vloss /= i + 1

        if VERBOSE:
            print(f"[{epoch + 1}] train loss: {running_loss:.3f}, test loss: {running_vloss:.3f}")

        if running_vloss < best_vloss:
            best_vloss = running_vloss
            epoch_number = epoch
            best_dict = model.state_dict()
            
    model.load_state_dict(best_dict)
    model.eval()
    
    
# -

cols = [
    'sentiment_sentiment_neg',
    'sentiment_sentiment_neu',
    'sentiment_sentiment_pos',
    'sentiment_sentiment_compound',
    'review_length',
]
extra['gm_predicted_prob'] = model(torch.Tensor(extra[cols].values)).detach().numpy()
extra['gm_predicted'] = extra['gm_predicted_prob'] > 0.5

extra.to_csv('predicted-gm-non-ebert.csv', index=False)




