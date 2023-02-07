import torch
import random
import numpy as np
from sklearn.metrics import r2_score


# reproducibility
def fix_random(seed: int) -> None:
    """Fix all the possible sources of randomness.

    Args:
        seed: the seed to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # slower


# training process
def train_model(mlp, loss_function, optimizer, epoch, trainloader, val_loader, device, writer):
    for e in range(epoch):
        current_loss = 0.0
        mlp.train()
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):

            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            targets = targets.reshape((targets.shape[0], 1))

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            #  Print statistics
            current_loss += loss.item()
           
        y_test, y_pred = test_model(mlp, val_loader, device)
        loss_val = loss_function(y_pred, y_test)
        writer.add_scalar("Loss/val", loss_val, e)
        
    
    r2 = r2_score(y_pred.detach().cpu().numpy().squeeze(), y_test.detach().cpu().numpy())
    
    return mlp, loss_val.item(), r2

# evaluation process
def test_model(mlp, data_loader,device):
    mlp.eval()
    y_pred = []
    y_test = []

    for data, targets in data_loader:
        data, targets = data.to(device), targets.to(device)
        y_pred.append(mlp(data))
        y_test.append(targets)

    y_test = torch.stack(y_test).squeeze()
    y_pred = torch.stack(y_pred).squeeze()

    return y_test, y_pred
