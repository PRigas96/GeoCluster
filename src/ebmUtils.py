import torch
import src.geometry as geo
from src.metrics import Linf
import numpy as np

def Reg(outputs, xlim, ylim):
    """
        Regularize the output of the projector inside the data-area
        
        Parameters:
            outputs (list): list of the outputs of the projector
            
        Returns:
            reg_cost (float): regularization cost
    """
    reg_cost = 0
    bbox = [xlim, ylim]

    for centroid in outputs:
        for centroid_dim in centroid:
            for boundary in bbox:
                current_prod = 1
                min_dist = np.inf
                for boundary_dim in boundary:
                    min_dist = min(min_dist, abs(boundary_dim - centroid_dim)) 
                    current_prod *= (boundary_dim - centroid_dim) / (abs(boundary_dim - centroid_dim))
                reg_cost += max(0, current_prod) * min_dist  

    return reg_cost

def RegLatent(latent):
    """
        Regularize the fuzziness of the latent variable

        Parameters:
            latent (list): list of the latent variables
            
        Returns:
            reg_cost (float): regularization cost

        #TODO check if regularization can be written in a more elegant way
    """
    kld = 0
    for i in range(latent.shape[1]):
        kld += torch.sum(torch.distributions.kl.kl_divergence(
            torch.distributions.normal.Normal(0, 1), 
            torch.distributions.normal.Normal(latent[i], 1)
        ))
        
    return kld

def loss_functional(y_hat, y_target, model):
    """
        Computes the loss functional of the model

        Parameters:
            y_hat (list): list of the outputs of the projector
            y_target (list): list of the target outputs
            model (LVGEBM): model
            
        Returns:
            loss (float): loss functional
    """
    n_centroids = y_hat.shape[0] # number of centroids
    n_data = y_target.shape[0] # number of data
    size = (n_data, n_centroids) # size of the loss matrix
    loss = torch.zeros(size) # initialize loss matrix
    for i in range(n_data):
        for j in range(n_centroids):
            # get square
            square = y_target[i] # get square
            # square = torch.tensor(square)
            #y_hat[j] = y_hat[j].clone().detach().requires_grad_(True)
            loss[i, j], _, _ = Linf(square, y_hat[j]) # compute loss
    return loss 