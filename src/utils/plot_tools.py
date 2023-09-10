import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import src.geometry as geo
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.signal import butter, filtfilt


def plot_data_on_manifold(fig,ax ,data, size=10, limits=[-10, 10, -10, 10]):
    """
        Plot the data points
        
        Parameters:
            fig: figure
            ax: axis
            data: data to be plotted
            size: size of the plot
            limits: limits of the plot
        
        Returns:
            None: Updates the figure, axis
    """
    patches = []
    num_polygons = data.__len__()

    for i in range(num_polygons):
        square = geo.create_square2(data[i])
        # print(square.shape)
        polygon = Polygon(square, True)
        patches.append(polygon)

    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)

    colors = 100*np.random.rand(len(patches))
    p.set_array(np.array(colors))

    ax.add_collection(p)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])

def AM_dem(signal, fc, fs=None, order=4, Ns=None):
    """
        Amplitude demodulation of the signal
        
        Parameters:
            signal: signal to be demodulated
            fc: carrier frequency
            fs: sampling frequency
            order: order of the filter
        
        Returns:
            demodulated signal
    """
    if Ns is not None:
        signal = signal[:Ns]
    if fs is None:
        fs = len(signal)
    # create the filter
    b, a = butter(order, fc/(fs/2), btype='lowpass')
    filtered_signal = filtfilt(b, a, signal)
    filtered_signal_norm = filtered_signal - signal
    # take the positive part and negative part
    positive_signal = filtered_signal_norm[filtered_signal_norm > 0]
    negative_signal = filtered_signal_norm[filtered_signal_norm < 0]
    positive_signal = np.repeat(positive_signal, 2) # repeat each sample twice
    positive_signal = positive_signal.tolist()
    # make it up to len(signal) samples
    diff_positive = len(signal) - len(positive_signal)
    if diff_positive >= 0:
        for i in range(diff_positive):
            positive_signal.append(positive_signal[-1]) 
    elif diff_positive < 0:
        for i in range(-diff_positive):
            positive_signal.pop()
    negative_signal = np.repeat(negative_signal, 2) # repeat each sample twice
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
    if len(filtered_signal)>=best_epoch:
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
        
        