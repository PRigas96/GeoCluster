import numpy as np
import math
import torch


def IsPointInsidePoly(vertex, poly_vertices):
    """
        Check if a point is inside a given polygon or not

        Parameters:
            vertex (list): [x,y] coordinates of the point
            poly_vertices (list): list of [x,y] coordinates of the polygon vertices

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

# given a square, we retrieve its 4 vertices
# the square is represented as a vector of
# center of mass, edge size and rotation

def create_square2(square):
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
    rotated_vertices = torch.mm(vertices - torch.tensor([x_center, y_center]), rotation_matrix.t()) + torch.tensor(
        [x_center, y_center])

    return rotated_vertices


def create_square2_np(square):
    """
        Create a square given its center of mass, edge size and rotation

        Parameters:
            square (list): [x_center, y_center, size, rotation]

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


# rotate a point in response to a reference point
def rotate_point(point, ref_point, theta):
    """
        Rotate a point in response to a reference point

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
        Check if two squares intersect

        Parameters:
            square1 (np.array): 4x2 array of the vertices of the square
            square2 (np.array): 4x2 array of the vertices of the square

        Returns:
            bool: True if the squares intersect
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
        Check if two squares intersect

        Parameters:
            square1 (np.array): 4x2 array of the vertices of the square
            square2 (np.array): 4x2 array of the vertices of the square

        Returns:
            bool: True if the squares intersect
    """

    for i in range(4):
        if check_if_intersect_segment_square(square1[i], square1[(i + 1) % 4], square2):
            return True
        if check_if_intersect(square1, square2):
            return True


def check_if_intersect_segment_square(point1, point2, square):
    """
        Check if a segment intersects a square

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
        Check if two segments intersect

        Parameters:
            point1 (list): [x,y] coordinates of the first point of the first segment
            point2 (list): [x,y] coordinates of the second point of the first segment
            point3 (list): [x,y] coordinates of the first point of the second segment
            point4 (list): [x,y] coordinates of the second point of the second segment

        Returns:
            bool: True if the segments intersect
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
        Check if a square intersects with any of the squares in data

        Parameters:
            data (list): list of np.array of the vertices of the squares
            x (float): x coordinate of the center of the square
            y (float): y coordinate of the center of the square
            size (float): size of the square
            rotation (float): rotation of the square in degrees

        Returns:
            bool: True if the square intersects with any of the squares in data
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