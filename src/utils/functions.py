import torch
import numpy as np


def NearestNeighbour(qp, sq, metric):
    """
        Finds the nearest neighbour of a query point in a list of data, using a given metric.

        Parameters:
            qp (torch.Tensor): the query point
            sq (torch.Tensor): the points to compare the query point to
            metric (callable): a metric function between a data object and a point

        Returns:
            (float, int):
                - the distance between the query point and its nearest neighbour, using the given metric
                - the index of the nearest neighbour
    """
    d_nn = np.inf
    z_nn = 0
    qp = torch.tensor(qp) if not torch.is_tensor(qp) else qp
    sq = sq if torch.is_tensor(sq) else torch.tensor(sq)
    for i, square in enumerate(sq):
        d_nn_sq = metric(square, qp)
        if d_nn_sq <= d_nn:
            d_nn = d_nn_sq
            z_nn = i
    return d_nn, z_nn
