import numpy as np
import random
import src.geometry as geo
# import sklearn
# from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


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
    ref = './data/squares/100/' + str(numberOfData)
    data = np.load(ref + 'sq_1_4.npy')
    data[:, -1] = np.deg2rad(data[:, -1])
    datapoints = []  # np.load(ref+'qp.npy')
    print("Data loaded.")

    return data, datapoints


def create_data(number_of_data, x0, size0, rotation0):
    data = []
    cnt = 1
    point_cnt = 0
    # number of acceptable intersections for each square generation
    max_num_of_collisions = 0.05 * number_of_data
    num_of_collisions = 0

    while cnt != number_of_data + 1:
        if (cnt >= len(x0)):
            break

        if (cnt == 1000) | (cnt == 5000) | (cnt == 10000) | (cnt == 25000) | (cnt == 35000) | (cnt == 45000) | (
                cnt == 50000):
            print("reached ", cnt, " squares!")
            np.save(f"./data/squares/100/{cnt}sq_1_4.npy", data)

        print("creating square ", cnt, "\n")
        x = x0[point_cnt][0]
        y = x0[point_cnt][1]
        point_cnt += 1
        size = random.choice(size0)
        rotation = random.choice(rotation0)

        current_square = np.array([x, y, size, rotation])

        # check if they intersect
        # if they do not intersect add them to the data
        if not geo.check_intersection(data, current_square):
            data.append(current_square)
            cnt += 1
            num_of_collisions = 0
        else:
            num_of_collisions += 1
            print(num_of_collisions)
            if num_of_collisions == max_num_of_collisions:
                print(
                    "max number of collisions reached! dataset creation terminates!\n"
                )
                break
    return data


def loadData_3d(numberOfData, numberOfQueryPoints):
    """
        Load data from data folder
    
        Parameters:
            numberOfData (int): number of data to load
            
        Returns:
            data (np.array): data

        Important:
            The data must be in the following format:
            [x0, y0, z0, w, h, d, theta, psi, phi] where:
            x0, y0, z0: coordinates of the center of the cuboid
            w, h, d: width, height and depth of the cuboid
            theta, psi, phi: rotation angles of the cuboid in rad for every axis
    """
    print("Loading data...")
    ref_data = './data_3d/10000cb/' + str(numberOfData) + 'cb_1_4.npy'
    ref_query_points = './data/squares/' + str(numberOfQueryPoints) + '/' + str(numberOfQueryPoints)
    data = np.load(ref_data)
    data[:, -1] = np.deg2rad(data[:, -1])
    datapoints = np.load(ref_query_points + 'qp.npy')
    print("Data loaded.")

    return data, datapoints


def create_data_3d(numberOfData, x0, width, height, depth, theta, psi, phi, cube=True, axis_aligned=True):
    """
        Create numberOfData data
    
        Parameters:
            numberOfData (int): number of data to load
            
        Returns:
            data (np.array): data

        Important:
            The data are in the following format:
            [x0, y0, z0, w, h, d, theta, phi] where:
            x0, y0: coordinates of the center of the cuboid
            w, h, z: width, height and depth of the cuboid
            theta: rotation angle of the square in rad
            phi: second
    """
    data = []
    point_cnt = 0
    cnt = 1  # Keep track of cuboids created so far

    # Maximum number of collisions allowed - If reached, the generation of cuboids stops
    maximum_num_of_collisions = 0.05 * numberOfData
    num_of_collisions = 0

    while cnt != numberOfData + 1:
        if (cnt >= len(x0)):
            break
        # Store the results for every 500 cuboids created, starting from 1.000 and ending to 10.000
        if 10000 <= cnt <= 50000 and (cnt - 10000) % 5000 == 0:
            print("reached ", cnt, " cuboids!")
            np.save(f"./data_3d/10000cb/{cnt}cb_1_4.npy", data)

        print("creating square ", cnt, "\n")
        # Create the square with random center of mass
        x = x0[point_cnt][0]
        y = x0[point_cnt][1]
        z = x0[point_cnt][2]
        point_cnt += 1

        # Check if the cuboid is actually a cube or not. Shape it accordingly
        if (cube):
            wid = hei = dep = random.choice(width)
        else:
            wid = random.choice(width)
            hei = random.choice(height)
            dep = random.choice(depth)

        if (axis_aligned):
            th = ps = ph = 0
        else:
            th = random.choice(theta)
            ps = random.choice(psi)
            ph = random.choice(phi)

        current_cuboid = np.array([x, y, z, wid, hei, dep, th, ps, ph])
        # Check if the current cuboid intersects with any of the cuboids in the data
        if not geo.check_intersection_3d(data, current_cuboid):
            data.append(current_cuboid)
            cnt += 1
            num_of_collisions = 0
        else:
            num_of_collisions += 1
            print("num of collisions is: ", num_of_collisions)
            # Terminate if maximum number of collisions reached
            if num_of_collisions == maximum_num_of_collisions:
                print(
                    "max number of collisions reached! dataset creation terminates!\n"
                )
                break
    return data


def create2moons4squares(**kwargs):
    """
        Create 2 moons for squares dataset

        Parameters:
            x_lim (list): x limits of the moons
            y_lim (list): y limits of the moons
            w_lim (list): width limits of the squares
            theta_lim (list): rotation limits of the squares
            plot (bool): plot the moons
            noise (float): noise of the moons
            numberOfData (int): number of data to create
            normalize (bool): normalize the data
            scale (bool): scale the data
            scale_factor (float): scale factor
            numberOfMoons (int): number of moons to create
            which_moon (str): which moon to create

        Returns:
            squares (np.array): squares
            
        Important:
            The squares are in the following format:
            [x0, y0, w, theta] where:
            x0, y0: coordinates of the center of the square
            w: width of the square
            theta: rotation angle of the square in rad
        
        Example:
            from src.utils.data import create2moons4squares 
            args = {
                'x_lim' : [0, 300],
                'y_lim' : [0, 300],
                'w_lim' : [1, 4],
                'theta_lim' : [1, 2],
                'plot' : True,
                'noise' : 0.1,
                'numberOfData': 200,
                'normalize' : True,
                'scale' : True,
                'scale_factor' : 0.1,
                'numberOfMoons' : 1,
                'which_moon' : 'upper',
            }
            data = create2moons4squares(**args)
    """
    x_lim = kwargs["x_lim"]  # [0, 300]
    y_lim = kwargs["y_lim"]  # [0, 300]
    w_lim = kwargs["w_lim"]  # [1,4]
    theta_lim = kwargs["theta_lim"]  # [0, 2pi]
    plot = kwargs["plot"]
    noise = kwargs["noise"]
    numberOfData = kwargs["numberOfData"]
    normalize = kwargs["normalize"]
    scale = kwargs["scale"]
    scale_factor = kwargs["scale_factor"]
    numberOfMoons = kwargs["numberOfMoons"]
    which_moon = kwargs["which_moon"]
    X, y = make_moons(n_samples=numberOfData, noise=noise)
    if numberOfMoons == 1:
        if which_moon == 'upper':
            X = X[y == 1]
        else:
            X = X[y == 0]
        numberOfData = X.shape[0]
    if normalize:
        x_min = X[:, 0].min()
        y_min = X[:, 1].min()
        X[:, 0] -= x_min
        X[:, 1] -= y_min
        if scale:
            X *= x_lim[1] * scale_factor
    if plot:
        if numberOfMoons == 1:
            plt.scatter(X[:, 0], X[:, 1])
            plt.show()
        else:
            plt.scatter(X[:, 0], X[:, 1], c=y)
            plt.show()
    squares = []

    def check_for_intersection(square, squares):
        for i in range(squares.__len__()):
            if square[0] < squares[i][0] + squares[i][2] and square[0] + square[2] > squares[i][0] and square[1] < \
                    squares[i][1] + squares[i][2] and square[1] + square[2] > squares[i][1]:
                return True
        return False

    for i in range(numberOfData - 1):
        x = int(X[i, 0])
        y = int(X[i, 1])
        w = np.random.randint(w_lim[0], w_lim[1])
        theta = np.random.randint(theta_lim[0], theta_lim[1])
        square = np.array([x, y, w, theta])
        if not check_for_intersection(square, squares):
            squares.append(square)
    squares = np.array(squares)
    if normalize:
        x_mean = squares[:, 0].mean().astype(int)
        y_mean = squares[:, 1].mean().astype(int)
        squares[:, 0] -= x_mean
        squares[:, 1] -= y_mean

    if plot:
        x_lim = [squares[:, 0].min(), squares[:, 0].max()]
        y_lim = [squares[:, 1].min(), squares[:, 1].max()]
        fig, ax = plt.subplots()
        for square in squares:
            ax.add_patch(
                plt.Rectangle((square[0], square[1]), square[2], square[2], angle=square[3], color='r', fill=False))

        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.show()
    # make degrees to radians
    squares[:, 3] = np.deg2rad(squares[:, 3])
    return squares


def createSquares(**kwargs):
    """
        Create squares for given data points
        
        Parameters:
            X (np.array): data points
            y (np.array): labels
            w_lim (list): width limits of the squares
            theta_lim (list): rotation limits of the squares
            numberOfData (int): number of data to create

        Returns:
            squares (np.array): squares
            
        Important:
            The squares are in the following format:
            [x0, y0, w, theta] where:
            x0, y0: coordinates of the center of the square
            w: width of the square
            theta: rotation angle of the square in rad
            
        Example:
            X, y = make_s_curve(n_samples=1000, noise=0.2)
            fig, ax = plt.subplots()
            ax.scatter(X[:, 0], X[:, 2], s=1, c=y)
            x_min = X[:, 0].min()
            y_min = X[:, 2].min()
            X[:, 0] -= x_min
            X[:, 2] -= y_min
            X = X[:, [0, 2]]
            X *= 300 * 0.5
            X = np.array(X)
            # plot X
            fig, ax = plt.subplots()
            ax.scatter(X[:, 0], X[:, 1], s=1, c=y)
            args = {
                'X': X,
                'y': y,
                'w_lim': [1,10],
                'theta_lim': [1, 3],
                'numberOfData': 1000
            }
            data = createSquares(**args)
            # plot
            fig, ax = plt.subplots()
            ax.scatter(data[:, 0], data[:, 1], s=1, c='b')
            print(data.shape)
            # plot rectangles
            import matplotlib.patches as patches
            for i in range(data.shape[0]):
                x = data[i, 0]
                y = data[i, 1]
                w = data[i, 2]
                theta = data[i, 3]
                rect = patches.Rectangle((x, y), w, w, angle=theta, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            plt.show()
    """
    X = kwargs['X']
    y = kwargs['y']
    w_lim = kwargs['w_lim']
    theta_lim = kwargs['theta_lim']
    numberOfData = kwargs['numberOfData']
    squares = []

    def check_for_intersection(square, squares):
        for i in range(squares.__len__()):
            if square[0] < squares[i][0] + squares[i][2] and square[0] + square[2] > squares[i][0] and square[1] < \
                    squares[i][1] + squares[i][2] and square[1] + square[2] > squares[i][1]:
                return True
        return False

    for i in range(numberOfData):
        x = int(X[i, 0])
        y = int(X[i, 1])
        w = np.random.randint(w_lim[0], w_lim[1])
        theta = np.random.randint(theta_lim[0], theta_lim[1])
        square = np.array([x, y, w, theta])
        if not check_for_intersection(square, squares):
            squares.append(square)
    squares = np.array(squares)
    x_mean = squares[:, 0].mean().astype(int)
    y_mean = squares[:, 1].mean().astype(int)
    squares[:, 0] -= x_mean
    squares[:, 1] -= y_mean
    squares[:, 3] = np.deg2rad(squares[:, 3])
    return squares
