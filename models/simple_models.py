import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, num_layers, dropout, nb_classes):
        super().__init__()
        ratio = (nb_classes / (3*input_size**2))**(1/num_layers)
        in_features = 3*input_size**2
        modules = []
        for i in range(num_layers - 1):
            out_features = int(ratio*in_features)
            modules.append(nn.Linear(in_features, out_features))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            in_features = out_features
        self.main = nn.Sequential(nn.Flatten(), *modules, nn.Linear(in_features, nb_classes))
    
    def forward(self, x):
        return self.main(x)
    