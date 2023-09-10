import torch
from torch import nn
from torch.nn.functional import one_hot
from torch.distributions import Dirichlet

class LVGEBM(nn.Module):
    """
        Latent Variable Generative Energy Based Model

        Parameters:
            n_centroids: number of centroids
            output_dim: output dimension
            latent_size: size of the latent space
            
        Args:
            z: latent space
            decoder: decoder
            projector: projector
            z_l: latent space without noise
            z_fuzzy: latent space with noise
            c: projected latent space
            c_fuzzy: projected latent space with noise
            
        Returns:
            forward: projected latent space
            sample_z: sample latent space
            col_one_hot: one hot encoding
            get_z: get latent space
        TODO: get pump to work in module 

    """
    def __init__(self, n_centroids, output_dim, latent_size=400):
        super(LVGEBM, self).__init__()
        self.n_centroids = n_centroids
        self.output_dim = output_dim
        self.latent_size = latent_size
        # decoder is a linear layer
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, output_dim, bias=False),
        )
        # init z to one-hot descrete latent variable
        self.z = nn.Parameter(
            Dirichlet(torch.ones(n_centroids)).sample((latent_size,)).T
            , requires_grad=True
        )
        self.projector = nn.Sequential(
            nn.Linear(n_centroids, n_centroids *10),
            nn.ReLU(),
            nn.Linear(n_centroids *10, n_centroids *100),
            nn.ReLU(),
            nn.Linear(n_centroids *100, n_centroids *10),
            nn.ReLU(),
            nn.Linear(n_centroids *10, n_centroids),
        )
        self.z_l = None
        self.z_fuzzy = None
        self.c = None
        self.c_fuzzy = None
            
        
    def col_one_hot(self, z):
        # one hot encoding
        #z = torch.argmax(z, dim=0)
        z = torch.argmin(z, dim=0)
        z = one_hot(z, num_classes=self.n_centroids).T
        if self.n_centroids == self.latent_size:
            # make z eye matrix
            z = torch.eye(self.n_centroids)

        return z.float()
    
    def get_z(self):
        return self.z
    
    # sample z from without one-hot encoding
    def sample_z(self, regularize=False):
        if regularize:
            z = self.z
        else:
            z = torch.bernoulli(self.z)
        y = self.decoder(z)
        return y, z

    def pump(self):
        self.c 
    
    def forward(self, z):
        z = self.col_one_hot(z)
        # make z dtype float
        y = self.decoder(z)
        self.z_l = y
        # make y fuzzy
        y_std = torch.randn(y.shape)* 1
        y_std = y_std.to(y.device)
        y = y + y_std
        self.z_fuzzy = y
        # project y_hat to space of y
        y_hat = self.projector(y.T).T
        self.c = y_hat

        return y_hat