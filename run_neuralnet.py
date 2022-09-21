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
from collections import Counter

from helpers import load_data, run_cv

# +
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X, y = load_data()

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

BATCH_SIZE = 16
train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, sampler=sampler)
test_dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)


# +
parameters = dict(
    input_dim = X.shape[1],  # FIXED
    output_dim = 1,  # FIXED
    hidden_dim = 16, # param
    dropout_rate = 0.25, # param
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
EPOCHS = 100

# to determine how many epochs are needed
epoch_number = 0
best_vloss = np.inf
best_dict = None

for epoch in range(EPOCHS):
    
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
    
    print(f"[{epoch + 1}] train loss: {running_loss:.3f}, test loss: {running_vloss:.3f}")
    
    if running_vloss < best_vloss:
        best_vloss = running_vloss
        epoch_number = epoch
        best_dict = model.state_dict()
        
print('Done!')
print(f"Best test loss: {best_vloss:.3f} at epoch {epoch_number}")


# +

best_model = NNet(
    parameters['input_dim'],
    parameters['hidden_dim'],
    parameters['output_dim'],
    parameters['dropout_rate'],
)
best_model.load_state_dict(best_dict)
best_model.eval()

print("Training F1-score:", f1_score(best_model(X_train).round().detach().numpy(), y_train))
print("Testing F1-score:", f1_score(best_model(X_test).round().detach().numpy(), y_test))

# -




