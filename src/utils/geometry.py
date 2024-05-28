import numpy as np
import math
import random  # if needed
import torch
import torch.nn.functional as F



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
    rotated_vertices = torch.mm(vertices - torch.tensor([x_center, y_center]), rotation_matrix.t()) + torch.tensor([x_center, y_center])

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
    x1 = x_center - (size/2)
    y1 = y_center - (size/2)
    x2 = x_center - (size/2)
    y2 = y_center + (size/2)
    x3 = x_center + (size/2)
    y3 = y_center + (size/2)
    x4 = x_center + (size/2)
    y4 = y_center - (size/2)

    vertice1 = [x1, y1]
    vertice2 = [x2, y2]
    vertice3 = [x3, y3]
    vertice4 = [x4, y4]

    # then rotate the vertices according to the given rotation
    # the reference point of the rotation is the center of mass of the square

    #We will not use rotation for now
    # vertice1 = rotate_point(vertice1, mass_center, rotation)
    # vertice2 = rotate_point(vertice2, mass_center, rotation)
    # vertice3 = rotate_point(vertice3, mass_center, rotation)
    # vertice4 = rotate_point(vertice4, mass_center, rotation)

    vertices = np.array([vertice1, vertice2, vertice3, vertice4])

    return vertices

def create_cuboid(cuboid):
    """
        Create a cuboid given its center of mass, edge sizes and rotations

        Parameters:
            square (list): [x_center, y_center, z_center, width, height, depth, theta, psi, phi]

        Returns:
            np.array: 8x3 array of the vertices of the cuboid
    """

    #Get the data of the cuboid
    x_center = cuboid[0]
    y_center = cuboid[1]
    z_center = cuboid[2]
    mass_center = np.array([x_center, y_center, z_center])
    width = cuboid[3]
    height = cuboid[4]
    depth = cuboid[5]
    theta = cuboid[6]
    psi = cuboid[7]
    phi = cuboid[8]

    # Calculate the coordinates of the vertices as if the cuboid is not rotated
    vertices = np.zeros((8, 3))
    offsets = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]])

    for i in range(8):
        vertices[i] = mass_center + np.multiply(offsets[i], [width / 2, height / 2, depth / 2])

    # Then rotate the vertices according to the given rotation - not needed
    # for i in range(8):
    #     vertices[i] = rotate_point3(vertices[i], mass_center, theta)
    #     vertices[i] = rotate_point3(vertices[i], mass_center, psi)
    #     vertices[i] = rotate_point3(vertices[i], mass_center, phi)

    return vertices

def rotate_point3(point, center, angle):
    """
    Rotate a point around a center by a specified angle.

    Parameters:
        point (list): The coordinates of the point [x, y, z]
        center (list): The coordinates of the center [x, y, z]
        angle (float): The angle of rotation in radians.

    Returns:
        np.array: The new coordinates of the point after rotation.
    """
    x_shifted = point[0] - center[0]
    y_shifted = point[1] - center[1]
    z_shifted = point[2] - center[2]

    x_new = x_shifted * math.cos(angle) - y_shifted * math.sin(angle) + center[0]
    y_new = x_shifted * math.sin(angle) + y_shifted * math.cos(angle) + center[1]
    z_new = z_shifted

    return [x_new, y_new, z_new]



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


# lets create a function to create a square with rotation


# def create_square2(x, y, size, rotation):
#     # create a square
#     square = np.array(
#         [[x, y], [x + size, y], [x + size, y + size], [x, y + size]])
#     # rotate the square
#     rotation = np.deg2rad(rotation)
#     square = np.array([rotate_point(square[0], square[i], rotation)
#                       for i in range(square.__len__())])
#     return square


# def rotate_point(init_point, point, theta):
#     if np.all(init_point == point):
#         return point
#     else:
#         x0 = init_point[0]
#         y0 = init_point[1]
#         x_ = point[0]
#         y_ = point[1]
#         r0 = np.arctan2(y_ - y0, x_ - x0)
#         r = theta
#         r_ = r0 + r

#         dist = np.sqrt((x0-x_)**2 + (y0-y_)**2)
#         #x = x0 + dist / np.sqrt(1 + np.tan(r_)**2)
#         x = x0 + dist * np.cos(r_)
#         y = y0 + np.sqrt(dist**2 - (x-x0)**2)
#         return [x, y]
# check if two squares intersect

# check if two squares intersect
def get_edges(vertices):
    """
        Generate edges from vertices of a cuboid

        Parameters:
            vertices (np.array): 8x3 array of the vertices of the cuboid

        Returns:
            list: List of edges defined by vertex indices
    """
    edges = []
    edges_set = set()
    for i in range(8):
        for j in range(i + 1, 8):
            edge = [i, j]
            edge.sort()
            edge_tuple = tuple(edge)
            if edge_tuple not in edges_set:
                edges.append(edge)
                edges_set.add(edge_tuple)
    return edges

def check_if_intersect3_simple(cuboid1, cuboid2):
    cube1_x = [vertex[0] for vertex in cuboid1]
    cube1_y = [vertex[1] for vertex in cuboid1]
    cube1_z = [vertex[2] for vertex in cuboid1]

    cube2_x = [vertex[0] for vertex in cuboid2]
    cube2_y = [vertex[1] for vertex in cuboid2]
    cube2_z = [vertex[2] for vertex in cuboid2]

    # Check for intersection along the x-axis
    if max(cube1_x) < min(cube2_x) or min(cube1_x) > max(cube2_x):
        return False

    # Check for intersection along the y-axis
    if max(cube1_y) < min(cube2_y) or min(cube1_y) > max(cube2_y):
        return False

    # Check for intersection along the z-axis
    if max(cube1_z) < min(cube2_z) or min(cube1_z) > max(cube2_z):
        return False

    # If no axis misalignment is found, cubes intersect
    return True

def check_if_intersect3(cuboid1, cuboid2):
    """
        Check if two 3D cuboids intersect

        Parameters:
            cuboid1 (np.array): 8x3 array of the vertices of the first cuboid
            cuboid2 (np.array): 8x3 array of the vertices of the second cuboid

        Returns:
            bool: True if the two cuboids intersect, False otherwise
    """
    edges1 = get_edges(cuboid1)
    edges2 = get_edges(cuboid2)
    for edge1 in edges1:
        for edge2 in edges2:
            if check_intersect(cuboid1[edge1[0]], cuboid1[edge1[1]], cuboid2[edge2[0]], cuboid2[edge2[1]]):
                return True
    return False

def check_intersect(p1, q1, p2, q2):
    """
        Check if two line segments intersect

        Parameters:
            p1 (np.array): starting point of the first line segment
            q1 (np.array): ending point of the first line segment
            p2 (np.array): starting point of the second line segment
            q2 (np.array): ending point of the second line segment

        Returns:
            bool: True if the two line segments intersect, False otherwise
    """
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        return q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False



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

def check_intersection_3d(data, cuboid):
    """
        Check if a cuboid intersects with any of the cuboids in data

        Parameters:
            data (list): list of np.array of the vertices of the cuboids
            x (float): x coordinate of the center of the cuboid
            y (float): y coordinate of the center of the cuboid
            size (float): size of the cuboid
            rotation (float): rotation of the cuboid in degrees

        Returns:
            bool: True if the cuboid intersects with any of the cuboids in data
    """
    # create a cuboid
    cuboid = create_cuboid(cuboid)
    # check if it intersects with any of the cuboids in the data
    for i in range(data.__len__()):
        # create a cuboid
        cuboid2 = create_cuboid(data[i])
        # check if they intersect
        if check_if_intersect3_simple(cuboid, cuboid2):
            return True
    return False