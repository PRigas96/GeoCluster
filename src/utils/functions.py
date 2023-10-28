import torch
import numpy as np

from src.ebmUtils import loss_functional
from src.metrics import Linf, Linf_array
from src.utils.data import loadData


# will sample points on the voronoi edges
# we will label them and fine tune a student network
# the sampler will be a module or a function?
def getUncertaintyArea(outputs, N, M, epsilon, x_area, y_area, model):
    """
        Sample N points in the area defined by x_area and y_area

        Parameters:
            outputs: the outputs of the teacher network
            N: number of points to sample
            M: number of points to return
            epsilon: the epsilon ball
            x_area: the x area to sample from
            y_area: the y area to sample from
            model: the teacher network

        Returns:
            m_points: the M points that are in the uncertainty area

        #TODO: Create more clever UN sampling
    """
    # first lets sample N points in the spaces defined by x_area and y_area
    n_points = torch.zeros(N ** 2, 2)
    x_p = np.linspace(x_area[0], x_area[1], N)
    y_p = np.linspace(y_area[0], y_area[1], N)
    for i in range(N ** 2):
        n_points[i] = torch.tensor([x_p[i % N], y_p[i // N]])
    # now lets get the uncertainty area for each point
    E = Linf_array(torch.tensor(n_points), torch.tensor(outputs))
    # get the min distance
    m_points = []
    m = 0
    i = 0
    flag = 1
    tmp = False
    print('Processing...')
    flag_temp = False
    while m <= M and i < N ** 2:
        E1 = E[i]
        F1 = E1.min()
        # diff should be E1 - F1 for all points (E1 is a vector ) and F1 is a scalar
        diff = torch.abs(E1 - F1)  # diff is a vector
        eps = epsilon * 300  # 300 is the max L_inf dist = a length measure of the space
        cnt = 0  # count the number of centroids that are close to the current point
        for j in range(E1.shape[0]):
            if diff[j] <= eps and F1 != E1[j]:
                cnt += 1
        if cnt == 1:
            # eps = eps * (1*F1/300 + 200/300) # new eps
            eps = eps * (1 / (F1 * 0.0001))
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


def getE(model, best_outputs, qp):
    """
        Get the Linf distance between the outputs and the qp points

        Parameters:
            model: the teacher network
            best_outputs: the outputs of the teacher network
            qp: the points to compare the outputs to

        Returns:
            F: the Linf distance between the outputs and the qp points
            z: the index of the qp point that is closest to the output
            F_sq: the Linf distance between the outputs and the sq points
            z_sq: the index of the sq point that is closest to the output
    """
    # get qp
    qp = torch.tensor(qp)
    # get outputs
    outputs = best_outputs
    outputs = torch.tensor(outputs)
    # make Linf between outputs points and qp (between them all)
    E = torch.zeros(outputs.shape[0], qp.shape[0])
    for i in range(outputs.shape[0]):
        for j in range(qp.shape[0]):
            E[i, j] = torch.max(torch.abs(outputs[i] - qp[j]))
    F, z = E.min(0)
    # now do the same for sq
    outputs = outputs.detach().numpy()
    sq_, _ = loadData(100)
    E_sq = loss_functional(outputs, sq_, model)
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
    for i, square in enumerate(sq):
        d_nn_sq, _, _ = Linf(square, qp)
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
        # find nn of qp[i]
        d_nn, z_nn = NearestNeighbour(qp[i], sq)
        # print("d_nn: ", d_nn)
        # print("z_nn: ", z_nn)
        # print("z[i]: ", z[i])
        # print("z_sq[z_nn]: ", z_sq[z_nn])
        if z_sq[z_nn] == z[i]:
            acc += 1
    return acc / qp.shape[0]
