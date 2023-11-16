import torch
import numpy as np

from src.ebmUtils import loss_functional
from src.metrics import Linf, Linf_3d, Linf_array, Linf_simple
from src.utils.data import loadData


"""
These are ebm_utils, getUncertaintyArea and getE. I think you should move them there instead of here.
"""
# will sample points on the voronoi edges
# we will label them and fine tune a student network
# the sampler will be a module or a function?
def getUncertaintyArea(outputs, N, M, epsilon, bounding_box):
    """
        Sample N points in the area defined by x_area and y_area

        Parameters:
            outputs: the outputs of the teacher network
            N: number of points to sample
            M: number of points to return
            epsilon: the epsilon ball
            bounding_box (list[list]): list of [min, max] limits for each coordinate

        Returns:
            m_points: the M points that are in the uncertainty area

        #TODO: Create more clever UN sampling
    """
    print("="*20)
    print("getUncertaintyArea")
    print(f'Ouputs are {outputs}')
    dim = outputs.shape[1]
    # first lets sample N points in the spaces defined by the bounding box.
    n_points = torch.zeros(N ** dim, dim)
    scale = torch.tensor([area[1] - area[0] for area in bounding_box])
    scale = torch.max(scale)
    spaces = [np.linspace(area[0], area[1], N) for area in bounding_box]
    print(f'scale is {scale}')
    for i in range(N ** dim):
        n_points[i] = torch.tensor([spaces[d][(i // (N ** d)) % N] for d in range(dim)])
    # plot n_points
 
    E = Linf_array(n_points, torch.tensor(outputs))
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
                ax.scatter(n_points[:, 0], n_points[:, 1], n_points[:, 2],c=z,cmap='viridis', marker='o', s = 1,alpha = 0.3)
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
    tmp = False
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
            # eps = eps * (1*F1/300 + 200/300) # new eps
            eps = eps * (1 / (F1 * 0.0001*(10*dim)))
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
            # eps = eps * 1/(0.01*F1) # new eps
            # eps = eps * (F1/300) * cnt
            eps = eps * 2
            # eps = eps * 2
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


def getE(model, best_outputs, qp, sq):
    """
        Get the Linf distance between the outputs and the qp points

        Parameters:
            model: the teacher network
            best_outputs: the outputs of the teacher network
            qp: the points to compare the outputs to
            sq: the sq points

        Returns:
            F: the Linf distance between the outputs and the qp points
            z: the index of the qp point that is closest to the output
            F_sq: the Linf distance between the outputs and the sq points
            z_sq: the index of the sq point that is closest to the output
    """
    # get qp
    qp = qp if torch.is_tensor(qp) else torch.tensor(qp)
    sq = sq if torch.is_tensor(sq) else torch.tensor(sq)
    # get outputs
    outputs = best_outputs
    outputs = outputs if torch.is_tensor(outputs) else torch.tensor(outputs)
    # make Linf between outputs points and qp (between them all)
    E = torch.zeros(outputs.shape[0], qp.shape[0])
    for i in range(outputs.shape[0]):
        for j in range(qp.shape[0]):
            E[i, j] = torch.max(torch.abs(outputs[i] - qp[j]))
    F, z = E.min(0)
    # now do the same for sq
    outputs = outputs.detach().numpy()
    E_sq = loss_functional(outputs, sq, model)
    F_sq, z_sq = E_sq.min(1)

    return F, z, F_sq, z_sq


def NearestNeighbour(qp, sq):
    """
        Find the nearest neighbour of qp in sq

        Parameters:
            qp: the query point
            sq: the points to compare the query point to

        Returns:
            d_nn: the Linf distance between the query point and its nearest neighbour
            z_nn: the index of the nearest neighbour
    """
    d_nn = np.inf
    z_nn = 0
    dim = qp.shape[0]
    qp = torch.tensor(qp) if not torch.is_tensor(qp) else qp
    sq = sq if torch.is_tensor(sq) else torch.tensor(sq)
    for i, square in enumerate(sq):
        if dim == 2:
            d_nn_sq = Linf_simple(square, qp)
        else:
            d_nn_sq = Linf_3d(square, qp)
        if d_nn_sq <= d_nn:
            d_nn = d_nn_sq
            z_nn = i
    return d_nn, z_nn


def Accuracy(F, z, F_sq, z_sq, qp, sq):
    """
        Get the accuracy of the model

        Parameters:
            F: the Linf distance between the outputs and the qp points
            z: the index of the qp point that is closest to the output
            F_sq: the Linf distance between the outputs and the sq points
            z_sq: the index of the sq point that is closest to the output
            qp: the points to compare the outputs to
            sq: the points to compare the outputs to

        Returns:
            acc: the accuracy of the model
    """
    acc = 0
    for i in range(qp.shape[0]):
        d_nn, z_nn = NearestNeighbour(qp[i], sq)
        if z_sq[z_nn] == z[i]:
            acc += 1
    return acc / qp.shape[0]
