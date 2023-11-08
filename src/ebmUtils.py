import torch
import src.geometry as geo
from src.metrics import Linf
import numpy as np
from torch import nn


def Reg(outputs, xlim, ylim, node_index, parent_node):
    """
        Regularize the output of the projector inside the data-area
        
        Parameters:
            outputs (list): list of the outputs of the projector
            
        Returns:
            reg_cost (float): regularization cost
    """
    #to alpha to proto einai to varos tis relu
    #meta oso katevaineis layers + 10 to alpha
    layer = len(node_index) - 1
    alpha = 10*layer # maybe 5 ?
    ce = nn.CrossEntropyLoss() 
    reg_cost = 0
    bbox = [xlim, ylim]

    if layer == 0:
        for centroid in outputs:
            for centroid_dim in centroid:
                for boundary in bbox:
                    current_prod = 1
                    min_dist = np.inf
                    for boundary_dim in boundary:
                        min_dist = min(min_dist, abs(boundary_dim - centroid_dim))
                        current_prod *= (boundary_dim - centroid_dim) / (abs(boundary_dim - centroid_dim))
                    reg_cost += max(0, current_prod) * min_dist
    else:
        """
            Regularize the output of the predictor inside the data-area
            
            reg_cost = sum of all previous regs in path + current
            current = alpha * ce(student(outputs), where do i  wanna belong to)
            -----------------------------------------------
            here you use parent_node.best_reg which is a constant. (i think so)
            We calculate all previous ones again, so that it goes where we want it.
            Otherwise, its just a constant loss.
            See pdf for more info.

        """

        child_label = int(node_index[-1])
        reg_cost = parent_node.best_reg + alpha * ce(parent_node.student(outputs),
                                                     torch.tensor([child_label for i in range(4)]))

    return reg_cost

def RegLatent(latent):
    """
        Regularize the fuzziness of the latent variable

        Parameters:
            latent (list): list of the latent variables
            
        Returns:
            reg_cost (float): regularization cost

        #TODO check if regularization can be written in a more elegant way
            like mu^2 = 0
            or a p-norm regularization
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