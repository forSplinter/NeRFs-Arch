from torch import nn
import torch.nn.functional as F

class DensityNerf(nn.Module):
    """
    1st Stage Basic Nerf Model (MLP): 
    approximate the density of a point in the scene based on its position

    Architecture : 
    (linear - relu)*hidden_size => (linear,linear)

    Args:
    - N : (int) paramater for positional encoding dimensionnality
    - hidden_size : (int[]) list representing the number of hidden layers and their size
    - out_feature_size : (int) : size of output features vector
    
    Returns:
    - density : (int) approximated density of the point
    - feature_vector : (float[]) vector to be used for 2nd stage Nerf to approximate color

    """
    def __init__(self, hidden_size, N, out_feature_size):
        super(DensityNerf, self).__init__()
        
        # Input Layer
        self.linear_input = nn.Linear(6*N, hidden_size[0])

        # Hidden Layers
        self.linears = nn.ModuleList()
        for i in range(len(hidden_size)- 1):
            self.linears.append(nn.Linear(hidden_size[i], hidden_size[i+1]))

        # Output Layers
        self.linear_density = nn.Linear(hidden_size[-1],1)
        self.linear_features = nn.Linear(hidden_size[-1], out_feature_size)

    def forward(self, x):
        density, feature_vector = None, None
        
        x = self.linear_input(x)
        x = F.relu(x)

        for l in self.linears :
            x = l(x)
            x = F.relu(x)

        density = self.linear_density(x)
        feature_vector = self.linear_features(x)

        return density, feature_vector
    
class ColorNerf(nn.Module):
    """
    2nd Stage Basic Nerf Model (MLP): 
    approximate the color of a point in the scene based on density feature vector and viewing direction

    Architecture : 
    (linear - relu)*hidden_size => linear
    Args:
    - N : (int) paramater for positional encoding dimensionnality
    - hidden_size : (int[]) list representing the number of hidden layers and their size
    - out_feature_size : (int) : size of output features vector
    
    Returns:
    - color : (int) approximated color of the point

    """
    def __init__(self, hidden_size, N, out_feature_size):
        super(ColorNerf, self).__init__()
        
        # Input Layer
        self.linear_input = nn.Linear(6*N + out_feature_size, hidden_size[0])

        # Hidden Layers
        self.linears = nn.ModuleList()
        for i in range(len(hidden_size)- 1):
            self.linears.append(nn.Linear(hidden_size[i], hidden_size[i+1]))

        # Output Layer
        self.linear_color = nn.Linear(hidden_size[-1],1)

    def forward(self, x):
        color = None
        
        x = self.linear_input(x)
        x = F.relu(x)

        for l in self.linears :
            x = l(x)
            x = F.relu(x)

        color = self.linear_color(x)

        return color
