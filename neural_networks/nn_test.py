import os
import numpy as np
import torch
from torch import nn
import pandas as pd
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from data_layer import MovieLensDataset
from utils import fix_random, train_model, test_model
from architecture import MLP

import itertools
from torch.utils.tensorboard import SummaryWriter

dataset = MovieLensDataset()

model = MLP(dataset.num_input, 16, 8)
model.load_state_dict(torch.load('/content/drive/MyDrive/01-DataAnalitycs/Progetto/neural_networks/best_models/dim168ep50bs8lr0.0001', map_location=torch.device('cpu')))


indices = np.arange(len(dataset.y))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

test_subset = Subset(dataset, test_idx)
test_loader = DataLoader(test_subset, batch_size=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_function = nn.MSELoss()

y_test, y_pred = test_model(model, test_loader, device)
loss_test = loss_function(y_pred, y_test)

print("\n --- Test Evaluation ---")
print( 'Test loss: {}'.format(loss_test.item()))
r2 = r2_score(y_pred.detach().cpu().numpy().squeeze(), y_test.detach().cpu().numpy())
print("Test R^2 Score : {:.2f}".format(r2))
