import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import src.geometry as geo
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection




def plot_data(obj, size=10, limits=[0,300,0,300]):
    """
        Plot the data points 

        Parameters:
            data: data to be plotted
            size: size of the plot
            limits: limits of the plot

        Returns:
            None
    """
    # turn data -1 element from deg to rad
    # copy obj to data
    data = np.copy(obj)
    data[:, -1] = np.rad2deg(data[:, -1])
    fig, ax = plt.subplots(figsize=(size, size))
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
    plt.show()

