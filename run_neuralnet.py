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

# +
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)
# -



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
# -

torch.tensor([c[int(t)]**-1 for t in y_train])

# +
from collections import Counter
c = Counter(int(x) for x in y_train)

samples_weight = torch.tensor([c[int(t)]**-1 for t in y_train])

sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, sampler=sampler)


# +
parameters = dict(
    input_dim = X.shape[1],  # FIXED
    output_dim = 1,  # FIXED
    hidden_dim = 8, # param
    dropout_rate = 0.05, # param
    learning_rate = 0.01, # param
    momentum = 0.9, # param
)

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
    
model = NNet(
    parameters['input_dim'],
    parameters['hidden_dim'],
    parameters['output_dim'],
    parameters['dropout_rate'],
)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=parameters['learning_rate'],
    momentum=parameters['momentum'],
)


# +

for epoch in range(100):  # loop over the dataset multiple times
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

    model.eval()
    f1_train = f1_score(model(X_train).round().detach().numpy(), y_train)
    f1_test = f1_score(model(X_test).round().detach().numpy(), y_test)
    
    print(f'[{epoch + 1}] train loss: {running_loss / (i + 1):.3f}; train f1-score: {f1_train:.3f}; test f1-score: {f1_test:.3f}')
        
print('Done!')

# +
model.eval()

f1_score(model(X_train).round().detach().numpy(), y_train)

# -

f1_score(model(X_test).round().detach().numpy(), y_test)








