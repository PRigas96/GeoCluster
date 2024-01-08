import numpy as np
import math


def create_cuboid(cuboid):
    """
        Create a cuboid given its center of mass, edge sizes and rotations

        Parameters:
            square (list): [x_center, y_center, z_center, width, height, depth, theta, psi, phi]

        Returns:
            np.array: 8x3 array of the vertices of the cuboid
    """

    # Get the data of the cuboid
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
    offsets = np.array(
        [[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]])

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
        return q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1],
                                                                                                               r[1])

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
