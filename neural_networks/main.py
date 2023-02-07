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



if __name__ == '__main__':

    # Set fixed random number seed
    seed = 42
    fix_random(seed)

     # look for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('mps')
    print("Device: {}".format(device))
   

    # Prepare  dataset
    dataset = MovieLensDataset()
    indices = np.arange(len(dataset.y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=seed)
    # create subsets and relative dataloader

    train_subset = Subset(dataset, train_idx)
    

    val_subset = Subset(dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=1)

    test_subset = Subset(dataset, test_idx)
    test_loader = DataLoader(test_subset, batch_size=1)


    # Define the loss function and optimizer
   
    hidden_sizes_1 = [8, 16, 32, 64]
    hidden_sizes_2 = [8, 16, 32, 64]
    nums_epochs = [5, 25, 50] 
    batch_sizes = [8, 16, 32]
    learning_rate = 1e-4
    hyperparameters = itertools.product(hidden_sizes_1, hidden_sizes_2, nums_epochs, batch_sizes)
    
    best_loss = 1
    best_r2 = 0
    # Run the training loop
    for hidden_sizes_1, hidden_sizes_2,num_epochs, batch in hyperparameters:
        log_name = "dim" + str(hidden_sizes_1) + str(hidden_sizes_2) + "ep" + str(num_epochs) + "bs" + str(batch) + "lr" + str(learning_rate)
        writer = SummaryWriter('runs/' + log_name)
        
        trainloader = DataLoader(train_subset, batch_size=batch, shuffle=True, num_workers=1)
       
        mlp = MLP(dataset.num_input, hidden_sizes_1, hidden_sizes_2,)
        mlp.to(device)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)
        
        mlp, loss, r2 = train_model(mlp, loss_function, optimizer, num_epochs, trainloader, val_loader, device, writer)

        print("Number of epochs: {} - Hidden size 1: {} - Hidden size 2: {} - Mini batch: {}".format(num_epochs, hidden_sizes_1, hidden_sizes_2, batch ))
        print( 'Val loss: {}'.format(loss))
        print("Train R^2 Score : {:.2f}".format(r2))
        
        if (loss < best_loss and r2 > best_r2):
            log_name_b = log_name
            best_loss = loss
            best_r2 = r2
            best_mlp = mlp
            num_epochs_b, hidden_sizes_1_b, hidden_sizes_2_b, batch_b = num_epochs, hidden_sizes_1, hidden_sizes_2, batch

        if not os.path.exists('/content/drive/MyDrive/01-DataAnalitycs/Progetto/neural_networks/models'):
            os.makedirs('/content/drive/MyDrive/01-DataAnalitycs/Progetto/neural_networks/models')
        torch.save(mlp.state_dict(), '/content/drive/MyDrive/01-DataAnalitycs/Progetto/neural_networks/models/' + log_name)

        writer.flush()
    writer.close()
    # Process is complete.
    print('Training process has finished.')

    print("--- Best Model ---")
    print("Number of epochs: {} - Hidden size 1: {} - Hidden size 2: {} - Mini batch: {}".format(num_epochs_b, hidden_sizes_1_b, hidden_sizes_2_b, batch_b ))
    print( 'Val loss: {}'.format(best_loss))
    print("Train R^2 Score : {:.2f}".format(best_r2))
    #salva modello best
    if not os.path.exists('/content/drive/MyDrive/01-DataAnalitycs/Progetto/neural_networks/best_models'):
            os.makedirs('/content/drive/MyDrive/01-DataAnalitycs/Progetto/neural_networks/best_models')
    torch.save(best_mlp.state_dict(), '/content/drive/MyDrive/01-DataAnalitycs/Progetto/neural_networks/best_models/' + log_name_b)
  

 
    #evaluation on test set
    y_test, y_pred = test_model(best_mlp, test_loader, device)
    loss_test = loss_function(y_pred, y_test)

    print("\n --- Test Evaluation ---")
    print( 'Test loss: {}'.format(loss_test.item()))
    r2 = r2_score(y_pred.detach().cpu().numpy().squeeze(), y_test.detach().cpu().numpy())
    print("Test R^2 Score : {:.2f}".format(r2))
    
    
#Later to restore: 
#model.load_state_dict(torch.load(filepath))