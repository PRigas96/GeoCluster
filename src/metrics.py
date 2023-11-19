# save here metric for tests
import numpy as np
import torch
from src import geometry as geo

# define Linf() function for a square and a point
def Linf(square, point):
    """
        Compute the Linf metric for a square and a point

        Parameters:
            square (np.array): square coordinates
            point (np.array): point coordinates

        Returns:
            min_dist (float): minimum distance between the point and the square
    """
    square = geo.create_square2(square)
    # 1st put coords around point and transform
    # in order to do the transform substract the point from each coord
    # normalize square to be around the point
    square_new = square_new[:, 0] - point[0], square_new[:, 1] - point[1]
    square_new_ = square_new[:, 0] - point[0], square_new[:, 1] - point[1]

    #square_new = np.array([np.subtract(square_point, point) for square_point in square])
    #square_new_ = np.array([np.subtract(square_point, point) for square_point in square])

    # find if y = abs(x) intersects the square and if so add the point to the list so that the clockwise order is preserved
    # find the intersection point
    # find if y = abs(x) intersects the square
    # for each edge of the square, check if y = abs(x) intersects the edge
    cnt = 0
    for i in range(square.shape[0]):
        # print("==================================")
        sq_pt1 = square_new[i]
        sq_pt2 = square_new[(i+1) % square.shape[0]]
        # alpha
        if not (sq_pt2[0] - sq_pt1[0] == 0):
            a = (sq_pt2[1] - sq_pt1[1])/(sq_pt2[0] - sq_pt1[0])
        else:
            continue
        # beta
        b = sq_pt1[1] - a*sq_pt1[0]
        # new point
        if not (a == 1 or a == -1):
            new_pts = np.array([[-b/(a-1), -b/(a-1)], [-b/(a+1), b/(a+1)]])
        else:
            continue
        # check if new points lie inside the segment
        # if so add to square_new
        for new_pt in new_pts:
            if (new_pt[0] >= min(sq_pt1[0], sq_pt2[0])) and (new_pt[0] <= max(sq_pt1[0], sq_pt2[0])) and \
                    (new_pt[1] >= min(sq_pt1[1], sq_pt2[1])) and (new_pt[1] <= max(sq_pt1[1], sq_pt2[1])):
                #print("new point is : ", new_pt)
                square_new_ = np.insert(square_new_, i+1+cnt, new_pt, axis=0)
                cnt += 1
    # get Linf from each point in square_new
    min_dist = np.inf
    for square_point in square_new_:
        
        dist = np.max(np.abs(square_point))
        if dist < min_dist:
            min_dist = dist
    return min_dist, square_new_, square_new


# define simplified Linf() function for a square and a point
def Linf_np(square, q_point):
    """
        Compute the simplified Linf metric for a square and a point

        Parameters:
            square (np.array): square coordinates  
            q_point (np.array): point coordinates

        Returns:
            min_dist (float): minimum distance between the point and the square

    """
    min_dist = np.inf
    for point in square:
        dist = np.max(np.abs(np.subtract(point, q_point)))
        if dist < min_dist:
            min_dist = dist
    return min_dist


def Linfp(x, y):
    return torch.max(torch.abs(x-y))


def Linf_array(q, c):
    e = torch.zeros((q.shape[0], c.shape[0]))
    for i in range(q.shape[0]):
        for j in range(c.shape[0]):
            e[i, j] = Linfp(q[i], c[j])
    return e

def Linf_3d(cuboid, point):
    dx = [0.0, 0.0]
    dy = [0.0, 0.0]
    dz = [0.0, 0.0]
    distance = 0.0
    min_dx, min_dy, min_dz = 0.0, 0.0, 0.0

    cuboid_vertices = geo.create_cuboid(cuboid)

    min_coords = np.min(cuboid_vertices, axis=0)
    max_coords = np.max(cuboid_vertices, axis=0)

    dx[0] = min_coords[0] - point[0]
    dx[1] = max_coords[0] - point[0]
    dy[0] = min_coords[1] - point[1]
    dy[1] = max_coords[1] - point[1]
    dz[0] = min_coords[2] - point[2]
    dz[1] = max_coords[2] - point[2]

    if dx[0] * dx[1] < 0:
        if dy[0] * dy[1] < 0:
            if dz[0] * dz[1] < 0:
                return -2
            else:
                distance = min(abs(dz[0]), abs(dz[1]))
        else:
            min_dy = min(abs(dy[0]), abs(dy[1]))
            min_dz = min(abs(dz[0]), abs(dz[1]))
            distance = min_dy if min_dy > min_dz else min_dz
    else:
        min_dx = min(abs(dx[0]), abs(dx[1]))
        min_dy = min(abs(dy[0]), abs(dy[1]))
        min_dz = min(abs(dz[0]), abs(dz[1]))
        distance = max(min_dx, min_dy, min_dz)

    return distance


def Linf_simple(square, q_point):
    """
        Compute the simplified Linf metric for a square and a point

        Parameters:
            square (torch.Tensor): square coordinates
            q_point (torch.Tensor): point coordinates

        Returns:
            min_dist (float): minimum distance between the point and the square, 0 if inside

    """
    dev = square.device
    square = geo.create_square2(square).to(dev)
    is_inside = geo.IsPointInsidePoly(q_point, square)

    if is_inside:
        return 0.0

    min_dist = float('inf')
    for point in square:

        dist = torch.max(torch.abs(point - q_point))
        if dist < min_dist:
            min_dist = dist
    return min_dist
