import numpy as np

def loadData(numberOfData):
    """
        Load data from data folder
    
        Parameters:
            numberOfData (int): number of data to load
            
        Returns:
            data (np.array): data

        Important:
            The data must be in the following format:
            [x0, y0, w, h, theta] where:
            x0, y0: coordinates of the center of the square
            w, h: width and height of the square
            theta: rotation angle of the square in rad
    """
    print("Loading data...")
    ref = './data/squares/' + str(numberOfData) +'/'+ str(numberOfData) 
    data = np.load(ref+'sq.npy')
    data[:, -1] = np.deg2rad(data[:, -1])
    datapoints = np.load(ref+'qp.npy')
    print("Data loaded.")

    return data, datapoints
    
