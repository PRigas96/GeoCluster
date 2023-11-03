import torch
import src.geometry as geo
from src.metrics import Linf

def Reg(outputs, xlim, ylim):
    """
        Regularize the output of the projector inside the data-area
        
        Parameters:
            outputs (list): list of the outputs of the projector
            
        Returns:
            reg_cost (float): regularization cost
    """
    reg_cost = 0

    for centroid in outputs:
        reg_cost += max(xlim[0] - centroid[0], 0) + max(centroid[0] - xlim[1], 0) + max(ylim[0] - centroid[1], 0) + max(centroid[1] - ylim[1], 0)
            
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