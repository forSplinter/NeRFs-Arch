from torch import nn
import torch
import torch.nn.functional as F

class NeRF(nn.Module):
    """
    Basic NeRF Model (MLP): 
    Approximate the density and color of a point in the scene based on its position and ray direction.

    Architecture : 
    1st Stage : position => (linear - relu)*hidden_density_size => (linear: density,linear: density features)
    2nd Stage : density features + ray direction => (linear - relu)*hidden_color_size => linear: color
    
    Args:
    - hidden_density_size : (int[]) Number of hidden layers for 1st Stage and sizes
    - hidden_color_size : (int[]) Number of hidden layers for 2nd Stage and sizes
    - N_pos : (int) paramater for positional encoding dimensionnality
    - N_dir : (int) paramater for viewing direction encoding dimensionnality
    - feature_density_size : (int) : dimension of output density features vector
    
    Returns:
    - density : (float) approximated density of the point
    - color: (R,G,B) int tuple of approximated color of the point
    """
    def __init__(self, hidden_density_size=[256]*8, hidden_color_size=[128], N_pos=10, N_dir=4, feature_density_size=256):
        super(NeRF, self).__init__()
        
        # Input Layers
        self.input_density = nn.Linear(6*N_pos, hidden_density_size[0])
        self.input_color = nn.Linear(6*N_dir + feature_density_size, hidden_color_size[0])

        # Hidden Layers
        self.density_stage = nn.ModuleList()
        for i in range(1,len(hidden_density_size)):
            self.density_stage.append(nn.Linear(hidden_density_size[i-1], hidden_density_size[i]))

        if len(hidden_color_size) > 1:
            self.color_stage = nn.ModuleList()
            for i in range(1,len(hidden_color_size)):
                self.color_stage.append(nn.Linear(hidden_color_size[i-1], hidden_color_size[i]))
        
        # Output Layers
        self.output_density = nn.Linear(hidden_density_size[-1],1)
        self.output_features = nn.Linear(hidden_density_size[-1], feature_density_size)
        self.output_color = nn.Linear(hidden_color_size[-1],3)

    def forward(self, x, y):
        """
        x : sample point position encoded
        y : ray direction encoded
        """
        if x.shape[0] != y.shape[0]:
            raise AssertionError("Number of point shoud be equal to number of direction vector")

        density, feature_vector, color = None, None, None
        
        # Input Density
        x = self.input_density(x)
        x = F.relu(x)

        for l in self.density_stage :
            x = l(x)
            x = F.relu(x)

        density = self.output_density(x)
        feature_vector = self.output_features(x)

        y = self.input_color(torch.cat([feature_vector, y], dim=-1))
        y = F.relu(y)

        if hasattr(self, 'color_stage') and len(self.color_stage) > 0:
            for l in self.color_stage:
                y = l(y)
                y = F.relu(y)
        
        color = self.output_color(y)

        return density, color