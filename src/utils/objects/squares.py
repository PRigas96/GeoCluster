import numpy as np
import math
import torch
import random
# import sklearn
# from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


def loadData(numberOfData):
    """
        Loads square objects from data folder.

        Parameters:
            numberOfData (int): number of data to load

        Returns:
            np.array: the loaded squares data

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
    print("Data loaded.")

    return data


def IsPointInsidePoly(vertex, poly_vertices):
    """
        Checks if a point is inside a given polygon or not.

        Parameters:
            vertex (list|np.array|torch.Tensor): [x,y] coordinates of the point
            poly_vertices (np.array|torch.Tensor): list of [x,y] coordinates of the polygon vertices

        Returns:
            bool: True if point is inside the polygon
    """
    n = poly_vertices.shape[0]
    result = False

    x = vertex[0]
    y = vertex[1]
    px1 = poly_vertices[0, 0]
    py1 = poly_vertices[0, 1]
    xints = x - 1.0
    for i in range(0, n + 1):
        px2 = poly_vertices[i % n, 0]
        py2 = poly_vertices[i % n, 1]
        if (min(py1, py2) < y):
            if (y <= max(py1, py2)):
                if (x <= max(px1, px2)):
                    if (py1 != py2):
                        xints = (y - py1) * (px2 - px1) / (py2 - py1) + px1
                    if (px1 == px2 or x <= xints):
                        result = not result
        px1 = px2
        py1 = py2

    return result


def create_square2(square):
    """
        Retrieves the 4 vertices of a given square.

        Parameters:
            square (list|np.array|torch.Tensor): the square represented as
                a vector of center of mass, edge size and rotation

        Returns:
            torch.Tensor: the 4 vertices of the given square
    """
    x_center, y_center, size, rotation = square

    size_half = size / 2

    # Coordinates of vertices without rotation
    vertices = torch.tensor([
        [x_center - size_half, y_center - size_half],
        [x_center - size_half, y_center + size_half],
        [x_center + size_half, y_center + size_half],
        [x_center + size_half, y_center - size_half]
    ])

    # Rotate the vertices according to the given rotation
    rotation = torch.tensor(rotation)
    rotation_matrix = torch.tensor([
        [torch.cos(rotation), -torch.sin(rotation)],
        [torch.sin(rotation), torch.cos(rotation)]
    ])

    # make all values from float->double
    rotated_vertices = torch.mm(vertices - torch.tensor([x_center, y_center]),
                                rotation_matrix.t()) + torch.tensor([x_center, y_center])

    return rotated_vertices


def create_square2_np(square):
    """
        Creates a square given its center of mass, edge size and rotation.

        Parameters:
            square (list|np.array): [x_center, y_center, size, rotation]
                x_center (float): x coordinate of the center of the square
                y_center (float): y coordinate of the center of the square
                size (float): size of the square
                rotation (float): rotation of the square in degrees

        Returns:
            np.array: 4x2 array of the vertices of the square
    """
    x_center = square[0]
    y_center = square[1]
    mass_center = np.array([x_center, y_center])
    size = square[2]
    rotation = square[3]

    # calculate the coordinates of the vertices as if the square is not rotated
    x1 = x_center - (size / 2)
    y1 = y_center - (size / 2)
    x2 = x_center - (size / 2)
    y2 = y_center + (size / 2)
    x3 = x_center + (size / 2)
    y3 = y_center + (size / 2)
    x4 = x_center + (size / 2)
    y4 = y_center - (size / 2)

    vertice1 = [x1, y1]
    vertice2 = [x2, y2]
    vertice3 = [x3, y3]
    vertice4 = [x4, y4]

    # then rotate the vertices according to the given rotation
    # the reference point of the rotation is the center of mass of the square

    # We will not use rotation for now
    # vertice1 = rotate_point(vertice1, mass_center, rotation)
    # vertice2 = rotate_point(vertice2, mass_center, rotation)
    # vertice3 = rotate_point(vertice3, mass_center, rotation)
    # vertice4 = rotate_point(vertice4, mass_center, rotation)

    vertices = np.array([vertice1, vertice2, vertice3, vertice4])

    return vertices


def rotate_point(point, ref_point, theta):
    """
        Rotates a point in response to a reference point.

        Parameters:
            point (list): [x,y] coordinates of the point
            ref_point (list): [x,y] coordinates of the reference point
            theta (float): rotation angle in degrees

        Returns:
            np.array: [x,y] coordinates of the rotated point
    """
    theta = np.deg2rad(theta)
    sin = math.sin(theta)
    cos = math.cos(theta)

    x = point[0]
    y = point[1]

    x_ref = ref_point[0]
    y_ref = ref_point[1]

    x = x - x_ref
    y = y - y_ref

    new_x = x * cos - y * sin
    new_y = x * sin + y * cos

    x = new_x + x_ref
    y = new_y + y_ref

    return np.array([x, y])


def check_if_intersect(square1, square2):
    """
        Checks if two squares intersect.

        Parameters:
            square1 (np.array): 4x2 array of the vertices of the square
            square2 (np.array): 4x2 array of the vertices of the square

        Returns:
            bool: True if the squares intersect, False otherwise
    """
    # check if any of the points of square1 are inside square2
    for i in range(4):
        if IsPointInsidePoly(square1[i], square2):
            return True
    # check if any of the points of square2 are inside square1
    for i in range(4):
        if IsPointInsidePoly(square2[i], square1):
            return True
    return False


def check_if_intersect2_simple(square1, square2):
    """
        Checks if two axis-aligned squares intersect.

        Parameters:
            square1 (np.array): 4x2 array of the vertices of the square
            square2 (np.array): 4x2 array of the vertices of the square

        Returns:
            bool: True if the squares intersect, False otherwise
    """
    square1_x = [vertex[0] for vertex in square1]
    square1_y = [vertex[1] for vertex in square1]

    square2_x = [vertex[0] for vertex in square2]
    square2_y = [vertex[1] for vertex in square2]

    # Check for intersection along the x-axis
    if max(square1_x) < min(square2_x) or min(square1_x) > max(square2_x):
        return False

    # Check for intersection along the y-axis
    if max(square1_y) < min(square2_y) or min(square1_y) > max(square2_y):
        return False

    # If no axis misalignment is found, cubes intersect
    return True


def check_if_intersect2(square1, square2):
    """
        Checks if two squares intersect.

        Parameters:
            square1 (np.array): 4x2 array of the vertices of the square
            square2 (np.array): 4x2 array of the vertices of the square

        Returns:
            bool: True if the squares intersect, False otherwise
    """
    for i in range(4):
        if check_if_intersect_segment_square(square1[i], square1[(i + 1) % 4], square2):
            return True
        if check_if_intersect(square1, square2):
            return True


def check_if_intersect_segment_square(point1, point2, square):
    """
        Checks if a segment intersects a square.

        Parameters:
            point1 (list): [x,y] coordinates of the first point of the segment
            point2 (list): [x,y] coordinates of the second point of the segment

        Returns:
            bool: True if the segment intersects the square
    """
    for i in range(4):
        if check_if_intersect_segment_segment(point1, point2, square[i], square[(i + 1) % 4]):
            return True
    return False


def check_if_intersect_segment_segment(point1, point2, point3, point4):
    """
        Checks if two segments intersect.

        Parameters:
            point1 (list): [x,y] coordinates of the first point of the first segment
            point2 (list): [x,y] coordinates of the second point of the first segment
            point3 (list): [x,y] coordinates of the first point of the second segment
            point4 (list): [x,y] coordinates of the second point of the second segment

        Returns:
            bool: True if the segments intersect, False otherwise
    """
    if (point2[0] - point1[0]) * (point4[1] - point3[1]) - (point2[1] - point1[1]) * (point4[0] - point3[0]) == 0:
        return False
    # find the intersection point
    t = ((point1[0] - point3[0]) * (point3[1] - point4[1]) - (point1[1] - point3[1]) * (point3[0] - point4[0])) / (
            (point1[0] - point2[0]) * (point3[1] - point4[1]) - (point1[1] - point2[1]) * (point3[0] - point4[0]))
    u = -((point1[0] - point2[0]) * (point1[1] - point3[1]) - (point1[1] - point2[1]) * (point1[0] - point3[0])) / (
            (point1[0] - point2[0]) * (point3[1] - point4[1]) - (point1[1] - point2[1]) * (point3[0] - point4[0]))
    # check if the intersection point is on the segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        return True
    return False


def check_intersection(data, square):
    """
        Checks if a square intersects with any of the squares in data.

        Parameters:
            data (np.array): list of squares in [x_center, y_center, size, rotation] format
            square (list|np.array): [x_center, y_center, size, rotation]
                x_center (float): x coordinate of the center of the square
                y_center (float): y coordinate of the center of the square
                size (float): size of the square
                rotation (float): rotation of the square in degrees

        Returns:
            bool: True if the square intersects with any of the squares in data, False otherwise
    """
    # create a square
    square = create_square2_np(square)
    # check if it intersects with any of the squares in the data
    for i in range(data.__len__()):
        # create a square
        square2 = create_square2_np(data[i])
        # check if they intersect
        if check_if_intersect2_simple(square, square2):
            return True
    return False


def create_data(number_of_data, x0, size0, rotation0):
    """
        Creates a list of non-overlapping squares from initial conditions.

        Parameters:
            number_of_data (int): number of data to generate
            x0 (list): list of [x, y] points to initiate the square center
            size0 (list): list of size values to pick randomly
            rotation0 (list): list of rotation values to pick randomly

        Returns:
            list: list of the generated non-overlapping squares
    """
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

        # check if the current square intersects any of the previously generated squares
        # if they do not intersect add it to the data
        if not check_intersection(data, current_square):
            data.append(current_square)
            cnt += 1
            num_of_collisions = 0
        else:
            num_of_collisions += 1
            print(num_of_collisions)
            if num_of_collisions == max_num_of_collisions:
                print("max number of collisions reached! dataset creation terminates!\n")
                break
    return data


def create2moons4squares(**kwargs):
    """
        Creates 2 moons for squares dataset.

        Keyword Parameters:
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
            np.array: squares

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
        Creates squares for given data points.

        Keyword Parameters:
            X (np.array): data points
            y (np.array): labels
            w_lim (list): width limits of the squares
            theta_lim (list): rotation limits of the squares
            numberOfData (int): number of data to create

        Returns:
            np.array: squares

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
