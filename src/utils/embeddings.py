import torch
import numpy as np
from torch import nn
from src.metrics import Linf_array


def Reg(centroids, bounding_box, node_index, parent_node):
    """
        Regularizes the output of the projector inside the data-area.

        Parameters:
            centroids (torch.Tensor): list of the centroids of the projector
            bounding_box (list[list]): list of [min, max] limits for each coordinate
            node_index (str): index of the node currently being used
            parent_node (Ktree.Node): parent node of the node currently being used

        Returns:
            float: regularization cost
    """
    layer = len(node_index) - 1
    alpha = 10 * layer  # maybe 5 ?
    ce = nn.CrossEntropyLoss()
    reg_cost = 0

    def Reg_adam(centroids, bbox):
        reg_cost = 0
        for centroid in centroids:
            for centroid_dim in centroid:
                for boundary in bbox:
                    current_prod = 1
                    min_dist = np.inf
                    for boundary_dim in boundary:
                        min_dist = min(min_dist, abs(boundary_dim - centroid_dim))
                        current_prod *= (boundary_dim - centroid_dim) / (abs(boundary_dim - centroid_dim))
                    reg_cost += max(0, current_prod) * min_dist
        return reg_cost

    if layer == 0:
        reg_cost = Reg_adam(centroids, bounding_box)
        return reg_cost
    else:
        reg_cost += Reg(centroids, bounding_box, node_index[:-1], parent_node.parent)
        reg_cost = 0.1 * layer * reg_cost
        child_label = torch.tensor([int(node_index[-1]) for _ in range(len(centroids))], dtype=torch.long)  # make tensor
        reg_cost += alpha * ce(parent_node.critic(centroids), child_label)
    return reg_cost


def RegLatent(latent):
    """
        Regularizes the fuzziness of latent variables.

        Parameters:
            latent (torch.Tensor): list of the latent variables

        Returns:
            float: regularization cost

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


def loss_functional(y_hat, y_target, metric):
    """
        Computes the loss functional of the model.

        Parameters:
            y_hat (torch.Tensor): list of the centroids of the projector
            y_target (torch.Tensor): list of the target centroids
            metric (callable): metric to use to compute the loss

        Returns:
            float: loss functional
    """
    n_centroids = y_hat.shape[0]  # number of centroids
    n_data = y_target.shape[0]  # number of data
    size = (n_data, n_centroids)  # size of the loss matrix
    loss = torch.zeros(size)  # initialize loss matrix
    for i in range(n_data):
        for j in range(n_centroids):
            loss[i, j] = metric(y_target[i], y_hat[j])  # compute loss
    return loss


# will sample points on the voronoi edges
# we will label them and fine tune a critic network
# the sampler will be a module or a function?
def getUncertaintyArea(centroids, N, M, epsilon, bounding_box):
    """
        Samples N points inside the data-area.

        Parameters:
            centroids (torch.Tensor): the centroids of the clustering network
            N (int): number of points to sample
            M (int): number of points to return
            epsilon (float): the epsilon ball
            bounding_box (list[list]): list of [min, max] limits for each coordinate

        Returns:
            list[list]: the M points that are in the uncertainty area
    """
    print("=" * 20)
    print("getUncertaintyArea")
    print(f'Centroids are {centroids}')
    dim = centroids.shape[1]
    # first lets sample N points in the spaces defined by the bounding box.
    n_points = torch.zeros(N ** dim, dim)
    scale = torch.tensor([area[1] - area[0] for area in bounding_box])
    scale = torch.max(scale)
    spaces = [np.linspace(area[0], area[1], N) for area in bounding_box]
    print(f'scale is {scale}')
    for i in range(N ** dim):
        n_points[i] = torch.tensor([spaces[d][(i // (N ** d)) % N] for d in range(dim)])
    # plot n_points

    E = Linf_array(n_points, torch.tensor(centroids))
    if dim == 3:
        # plot dx, dy, dz in n_points
        dx = n_points[:, 0].max() - n_points[:, 0].min()
        dy = n_points[:, 1].max() - n_points[:, 1].min()
        dz = n_points[:, 2].max() - n_points[:, 2].min()
        print(f'dx is {dx}, dy is {dy}, dz is {dz}')
        # now lets get the uncertainty area for each point
        z = E.argmin(1)
        print(f'z is {z.shape}')
        debug = False
        if debug:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(n_points[:, 0], n_points[:, 1], n_points[:, 2], c=z, cmap='viridis', marker='o', s=1,
                           alpha=0.3)
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                plt.show()
            except ImportError:
                print("Matplotlib not installed. Cannot plot.")
                return
    # get the min distance
    m_points = []
    m = 0
    i = 0
    flag = 1
    print('Processing...')
    flag_temp = False
    while m <= M and i < N ** dim:
        E1 = E[i]
        F1 = E1.min()
        # diff should be E1 - F1 for all points (E1 is a vector ) and F1 is a scalar
        diff = torch.abs(E1 - F1)  # diff is a vector
        eps = epsilon * scale  # 300 is the max L_inf dist = a length measure of the space
        cnt = 0  # count the number of centroids that are close to the current point
        for j in range(E1.shape[0]):
            if diff[j] <= eps and F1 != E1[j]:
                cnt += 1
        if cnt == 1:
            eps = eps * (1 / (F1 * 0.0001 * (10 * dim)))
            # only 2 centroids are close => 1/F1
            tmp = False
            for j in range(E1.shape[0]):
                if diff[j] <= eps and F1 != E1[j]:
                    tmp = True
                    m_points.append(n_points[i])
                    m += 1
                    i += 1
                    break
            if tmp == False:
                flag_temp = True

        elif cnt > 1:
            eps = eps * 2
            tmp = False
            for j in range(E1.shape[0]):
                if diff[j] <= eps and F1 != E1[j]:
                    m_points.append(n_points[i])
                    tmp = True
                    m += 1
                    i += 1
                    break
            if tmp != True:
                flag_temp = True
        else:
            flag += 1
            i += 1
            continue
        if flag_temp == True:
            flag_temp = False
            i += 1
            flag += 1

    print(f'flag is {flag}')
    print(f'm is {m}')
    print(f'i is {i}')
    return m_points
