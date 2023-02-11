import torch
from torch import nn


# definition of a Multi Layer Perceptron with 2 hidden layers and with ReLu as activation function
class MLP(nn.Module):
    #Multilayer Perceptron for regression.

    def __init__(self, input_size,hidden_sizes_1, hidden_sizes_2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes_1),
            nn.ReLU(),
            nn.Linear(hidden_sizes_1,hidden_sizes_2),
            nn.ReLU(),
            nn.Linear(hidden_sizes_2, 1)
        )

    def forward(self, x):
        #Forward pass
        return self.layers(x)

# definition of a Multi Layer Perceptron with 2 hidden layers and with ReLu as activation function and with dropout
class MLP_DropOut(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, dropout_prob, depth):
        super(MLP_DropOut, self).__init__()

        self.input_size = input_size
        self.dp = dropout_prob

        model = [
            nn.Linear(input_size, hidden_size_1),
            nn.BatchNorm1d(hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.BatchNorm1d(hidden_size_2),
            nn.ReLU()
        ]

        for i in range(depth):
            if dropout_prob > 0:
                model.append(nn.Dropout(self.dp))

            model += [
                nn.Linear(hidden_size_2, hidden_size_1),
                nn.BatchNorm1d(hidden_size_1),
                nn.ReLU(),
                nn.Linear(hidden_size_1, hidden_size_2),
                nn.BatchNorm1d(hidden_size_2),
                nn.ReLU()
            ]

        self.model = nn.Sequential(*model)

        self.output = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        h = self.model(x)
        out = self.output(h)
        return out
