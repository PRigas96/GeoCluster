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
    square_new = np.array([np.subtract(square_point, point)
                          for square_point in square])
    square_new_ = np.array([np.subtract(square_point, point)
                           for square_point in square])

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
def Linf_(square, q_point):
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
