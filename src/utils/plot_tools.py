import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from src.utils.objects import squares
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.signal import butter, filtfilt
from src.utils.embeddings import loss_functional


def plot_data(obj, size=10, limits=[0, 300, 0, 300]):
    """
        Plots given data objects.

        Parameters:
            obj (torch.Tensor): data objects to be plotted
            size (int): size of the plot
            limits (list): limits of the plot
    """
    # turn data -1 element from deg to rad
    # copy obj to data
    data = np.copy(obj)
    data[:, -1] = np.rad2deg(data[:, -1])
    fig, ax = plt.subplots(figsize=(size, size))
    patches = []
    num_polygons = data.__len__()

    for i in range(num_polygons):
        square = squares.create_square2(data[i])
        # print(square.shape)
        polygon = Polygon(square, True)
        patches.append(polygon)

    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)

    colors = 100 * np.random.rand(len(patches))
    p.set_array(np.array(colors))

    ax.add_collection(p)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    plt.show()


def plot_data_on_manifold(fig, ax, obj, size=10, limits=[-10, 10, -10, 10]):
    """
        Plots the data points on a manifold.
        
        Parameters:
            fig: figure
            ax: axis
            obj (torch.Tensor): data objects to be plotted
            size (int): size of the plot
            limits (list): limits of the plot
    """
    data = np.copy(obj)
    data[:, -1] = np.rad2deg(data[:, -1])
    patches = []
    num_polygons = data.__len__()

    for i in range(num_polygons):
        # make float data[i] -> double
        for j in range(data[i].__len__()):
            data[i][j] = data[i][j].astype(np.double)
        square = squares.create_square2_np(data[i])
        # print(square.shape)
        polygon = Polygon(square, True)
        patches.append(polygon)

    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)

    colors = 100 * np.random.rand(len(patches))
    p.set_array(np.array(colors))

    ax.add_collection(p)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])


def AM_dem(signal, fc, fs=None, order=4, Ns=None):
    """
        Amplitude demodulation of the signal.
        
        Parameters:
            signal (np.array): signal to be demodulated
            fc (float): carrier frequency
            fs (int|None): sampling frequency
            order (int): order of the filter
            Ns (int|None): number of signal inputs to use
        
        Returns:
            (np.array, np.array, np.array): the demodulated signal
    """
    if Ns is not None:
        signal = signal[:Ns]
    if fs is None:
        fs = len(signal)
    # create the filter
    b, a = butter(order, fc / (fs / 2), btype='lowpass')
    filtered_signal = filtfilt(b, a, signal)
    filtered_signal_norm = filtered_signal - signal
    # take the positive part and negative part
    positive_signal = filtered_signal_norm[filtered_signal_norm > 0]
    negative_signal = filtered_signal_norm[filtered_signal_norm < 0]
    positive_signal = np.repeat(positive_signal, 2)  # repeat each sample twice
    positive_signal = positive_signal.tolist()
    # make it up to len(signal) samples
    diff_positive = len(signal) - len(positive_signal)
    if diff_positive >= 0:
        for i in range(diff_positive):
            positive_signal.append(positive_signal[-1])
    elif diff_positive < 0:
        for i in range(-diff_positive):
            positive_signal.pop()
    negative_signal = np.repeat(negative_signal, 2)  # repeat each sample twice
    negative_signal = negative_signal.tolist()
    # make it up to len(signal) samples
    diff_negative = len(signal) - len(negative_signal)
    if diff_negative >= 0:
        for i in range(diff_negative):
            negative_signal.append(negative_signal[-1])
    elif diff_negative < 0:
        for i in range(-diff_negative):
            negative_signal.pop()
    upper_signal = positive_signal + filtered_signal
    lower_signal = negative_signal + filtered_signal

    return upper_signal, lower_signal, filtered_signal


def plot_AM_dem(upper_signal, lower_signal, filtered_signal, signal, best_epoch):
    """
        Plots the demodulated signal.
        
        Parameters:
            upper_signal (np.array): upper signal
            lower_signal (np.array): lower signal
            filtered_signal (np.array): filtered signal
            signal (np.array): original signal
            best_epoch (int): best epoch
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(range(filtered_signal.shape[0]),
               filtered_signal,
               s=1,
               c='royalblue',
               alpha=0.5,
               marker='*'
               )
    ax.fill_between(range(filtered_signal.shape[0]),
                    lower_signal,
                    upper_signal,
                    color='royalblue',
                    alpha=0.4,
                    label='$Loss\ Functional$'
                    )
    if len(filtered_signal) >= best_epoch:
        label = '$Best\ Epoch\ at\ LF={:.4f}$'.format(signal[best_epoch])
        ax.scatter(best_epoch,
                   signal[best_epoch],
                   s=100,
                   c='red',
                   alpha=1,
                   marker='*',
                   label=label
                   )
    ax.set_xlabel('$Epochs$')
    ax.legend()
    plt.show()


def createManifold(model
                   , y_pred
                   , metric
                   , x_discr=100
                   , y_discr=100
                   , x_lim=[0, 300]
                   , y_lim=[0, 300]
                   ):
    """
        Creates a manifold.
        
        Parameters:
            model: model to be used
            y_pred (torch.Tensor): predicted outputs
            metric (callable): the metric to use for the loss function
            x_discr (int): x discretization
            y_discr (int): y discretization
            x_lim (list): x limits
            y_lim (list): y limits
            
        Returns:
            manifold: manifold
    """
    manifold = torch.zeros((x_discr, y_discr, 4))
    for i, x in enumerate(np.linspace(x_lim[0], x_lim[1], x_discr)):
        for j, y in enumerate(np.linspace(y_lim[0], y_lim[1], y_discr)):
            manifold[i, j, :] = torch.Tensor([x, y, 0, 0])
    manifold = manifold.view(-1, 4)  # flatten
    points = manifold[:, :2]  # get points (x,y)
    points = torch.cat((points, torch.zeros((points.shape[0], 2))), dim=1)  # add zeros for z and w
    outputs = y_pred
    points = points
    cost = loss_functional(outputs, points, metric)  # calculate cost
    F, z = cost.min(1)  # get energy and latent
    for i in range(manifold.shape[0]):
        manifold[i, -2] = F[i]
        manifold[i, -1] = z[i]
    manifold = manifold.reshape(x_discr, y_discr, 4)  # reshape
    return manifold


def plotManifold(data, manifold, best_outputs, x_lim, y_lim, dim='2D'):
    """
        Plot the manifold
        
        Parameters:
            data: data to be plotted
            manifold: manifold to be plotted
            best_outputs: best outputs
            x_lim: x limits
            y_lim: y limits
            dim: dimension
            
        Returns:
            None: plots the manifold
    """
    fig = plt.figure(figsize=(10, 10))

    if dim == '2D':
        ax = fig.add_subplot(111)
    elif dim == '3D':
        ax = fig.add_subplot(111, projection='3d')
    level = 200
    ax.contourf(manifold[:, :, 0], manifold[:, :, 1], manifold[:, :, -2], level, cmap='viridis', alpha=0.9)
    o = best_outputs.detach().numpy()
    no = o.shape[0]
    lb = np.linspace(0, no - 1, no)
    ax.scatter(o[:, 0], o[:, 1], c=lb, s=50)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    # vis data
    plot_data_on_manifold(fig, ax, data, size=10, limits=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]])
