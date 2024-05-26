# save here metric for tests
import numpy as np
import torch
from src.utils.objects import squares, cuboids, ellipses


# define Linf() function for a square and a point
def Linf(square, point):
    """
        Computes the L_inf metric for a square and a point.

        Parameters:
            square (np.array): square vector representation
            point (np.array): point coordinates

        Returns:
            float: distance between the point and the square
    """
    square = squares.create_square2(square)
    # 1st put coords around point and transform
    # in order to do the transform substract the point from each coord
    # normalize square to be around the point
    square_new = square_new[:, 0] - point[0], square_new[:, 1] - point[1]
    square_new_ = square_new[:, 0] - point[0], square_new[:, 1] - point[1]

    # square_new = np.array([np.subtract(square_point, point) for square_point in square])
    # square_new_ = np.array([np.subtract(square_point, point) for square_point in square])

    # find if y = abs(x) intersects the square and if so add the point to the list so that the clockwise order is preserved
    # find the intersection point
    # find if y = abs(x) intersects the square
    # for each edge of the square, check if y = abs(x) intersects the edge
    cnt = 0
    for i in range(square.shape[0]):
        # print("==================================")
        sq_pt1 = square_new[i]
        sq_pt2 = square_new[(i + 1) % square.shape[0]]
        # alpha
        if not (sq_pt2[0] - sq_pt1[0] == 0):
            a = (sq_pt2[1] - sq_pt1[1]) / (sq_pt2[0] - sq_pt1[0])
        else:
            continue
        # beta
        b = sq_pt1[1] - a * sq_pt1[0]
        # new point
        if not (a == 1 or a == -1):
            new_pts = np.array([[-b / (a - 1), -b / (a - 1)], [-b / (a + 1), b / (a + 1)]])
        else:
            continue
        # check if new points lie inside the segment
        # if so add to square_new
        for new_pt in new_pts:
            if (new_pt[0] >= min(sq_pt1[0], sq_pt2[0])) and (new_pt[0] <= max(sq_pt1[0], sq_pt2[0])) and \
                    (new_pt[1] >= min(sq_pt1[1], sq_pt2[1])) and (new_pt[1] <= max(sq_pt1[1], sq_pt2[1])):
                # print("new point is : ", new_pt)
                square_new_ = np.insert(square_new_, i + 1 + cnt, new_pt, axis=0)
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
        Computes the simplified L_inf metric for a square and a point.

        Parameters:
            square (np.array): square vector representation
            q_point (np.array): point coordinates

        Returns:
            float: distance between the square and the point
    """
    min_dist = np.inf
    for point in square:
        dist = np.max(np.abs(np.subtract(point, q_point)))
        if dist < min_dist:
            min_dist = dist
    return min_dist


def Linfp(x, y):
    """
        Computes the L_inf distance between two points.
        Parameters:
            x (np.array|torch.Tensor): first point
            y (np.array|torch.Tensor): second point

        Returns:
            float: distance between the two points
    """
    return torch.max(torch.abs(x - y))


def Linf_array(q, c):
    """
    Computes a list of distances between two lists of points.
    Parameters:
        q (np.array|torch.Tensor): first list of points
        c (np.array|torch.Tensor): second list of points

    Returns:
        torch.Tensor: tensor of size (len(q), len(c)) containing
            all pairs of distances between the two given lists of points
    """
    e = torch.zeros((q.shape[0], c.shape[0]))
    for i in range(q.shape[0]):
        for j in range(c.shape[0]):
            e[i, j] = Linfp(q[i], c[j])
    return e


def Linf_3d(cuboid, point):
    """
        Computes the L_inf metric for a cuboid and a point.

        Parameters:
            cuboid (np.array|torch.Tensor): cuboid vector representation
            point (np.array|torch.Tensor): point coordinates

        Returns:
            float: distance between the cuboid and the point,
                -2 if the point is inside the cuboid
    """
    cuboid = cuboid if not torch.is_tensor(cuboid) else cuboid.detach().numpy()
    point = point if not torch.is_tensor(point) else point.detach().numpy()

    dx = [0.0, 0.0]
    dy = [0.0, 0.0]
    dz = [0.0, 0.0]

    cuboid_vertices = cuboids.create_cuboid(cuboid)

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
            float: distance between the square and the point, 0 if inside
    """
    dev = square.device
    square = squares.create_square2(square).to(dev)
    is_inside = squares.IsPointInsidePoly(q_point, square)

    if is_inside:
        return 0.0

    min_dist = float('inf')
    for point in square:

        dist = torch.max(torch.abs(point - q_point))
        if dist < min_dist:
            min_dist = dist
    return min_dist


def distance_ellipse_2_point(ellipse, point):
    """
    Computes the distance between an ellipse and a point.
    Parameters:
        ellipse (np.array|torch.Tensor): ellipse vector representation
        point (np.array|torch.Tensor): point coordinates

    Returns:
        float: distance between the ellipse and the point
    """
    ellipse = ellipse if not torch.is_tensor(ellipse) else ellipse.detach().numpy()
    point = point if not torch.is_tensor(point) else point.detach().numpy()
    a = ellipse[0]
    b = ellipse[1]
    ellipse_center_x = ellipse[2]
    ellipse_center_y = ellipse[3]
    p_x = point[0] - ellipse_center_x
    p_y = point[1] - ellipse_center_y
    a2 = pow(a, 2)
    b2 = pow(b, 2)
    k = a2 - b2
    coeffs = [0, 0, 0, 0, 0]
    coeffs[4] = - pow(a, 6) * pow(p_x, 2)
    coeffs[3] = 2 * pow(a, 4) * p_x * k
    coeffs[2] = (pow(a, 4) * pow(p_x, 2) + pow(a, 2) * pow(b, 2) * pow(p_y, 2)
                 - pow(a, 2) * pow(k, 2))
    coeffs[1] = -2 * pow(a, 2) * p_x * k
    coeffs[0] = pow(k, 2)
    # print(coeffs)
    roots = np.roots(coeffs)
    # print(roots)
    xs = roots[np.isreal(roots)].real
    ys = [(pow(b, 2) * p_y * xs[0]) / (-k * xs[0] + pow(a, 2) * p_x),
          (pow(b, 2) * p_y * xs[1]) / (-k * xs[1] + pow(a, 2) * p_x)]
    # print(x1, y1)
    # print(x2, y2)
    # plt.plot(xs[0] + ellipse_center_x, ys[0] + ellipse_center_y, marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
    # plt.plot(xs[1] + ellipse_center_x, ys[1] + ellipse_center_y, marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
    # plt.plot(xs[0], ys[0], marker="o", markersize=5, markeredgecolor="green",
    #          markerfacecolor="green")
    # plt.plot(xs[1], ys[1], marker="o", markersize=5, markeredgecolor="green",
    #          markerfacecolor="green")
    # plt.plot(point[0], point[1], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
    # plt.plot(p_x, p_y, marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue")
    # print(xs, ys, ellipse_center_x, ellipse_center_y, p_x, p_y)
    return min(ellipses.distance_between_points([xs[0], ys[0]], [p_x, p_y]),
               ellipses.distance_between_points([xs[1], ys[1]], [p_x, p_y]))



def compute_distances_2d(segments, centroid):
    # print devices of segments and of centroids and check if equal
    # print(segments.device, centroid.device)
    if len(segments.shape) == 1:
        segments = segments.unsqueeze(0)
    # Unpack the centroid coordinates
    cx, cy = centroid
    # Unpack the segments
    x0s, y0s, lengths, thetas = (
        segments[:, 0],
        segments[:, 1],
        segments[:, 2],
        segments[:, 3],
    )
    # Calculate the endpoints of the segments
    x1s = x0s + lengths * torch.cos(thetas)
    y1s = y0s + lengths * torch.sin(thetas)
    # Vector from (x0, y0) to centroid
    vec_p0_c = torch.stack([cx - x0s, cy - y0s], dim=1)
    # Direction vector of the segments
    vec_p0_p1 = torch.stack([x1s - x0s, y1s - y0s], dim=1)
    # Projection scalar of vec_p0_c onto vec_p0_p1
    dot_products = torch.sum(vec_p0_c * vec_p0_p1, dim=1)
    segment_lengths_squared = torch.sum(vec_p0_p1 * vec_p0_p1, dim=1)
    projection_scalars = dot_products / segment_lengths_squared
    # Clamp the projection_scalars to lie within the segment
    projection_scalars = torch.clamp(projection_scalars, min=0, max=1)
    # Calculate the nearest points on the segments to the centroid
    nearest_xs = x0s + projection_scalars * (x1s - x0s)
    nearest_ys = y0s + projection_scalars * (y1s - y0s)
    # Distance from nearest points on the segments to the centroid
    distances = torch.sqrt((nearest_xs - cx) ** 2 + (nearest_ys - cy) ** 2)

    return distances


def compute_distances_3d(segments, centroid):
    if len(segments.shape) == 1:
        segments = segments.unsqueeze(0)
    # Unpack the centroid coordinates
    cx, cy, cz = centroid
    # Unpack the segments
    x0s, y0s, z0s, lengths, thetas, phis = (
        segments[:, 0],
        segments[:, 1],
        segments[:, 2],
        segments[:, 3],
        segments[:, 4],
        segments[:, 5],
    )
    # Calculate the endpoints of the segments
    x1s = x0s + lengths * torch.sin(thetas) * torch.cos(phis)
    y1s = y0s + lengths * torch.sin(thetas) * torch.sin(phis)
    z1s = z0s + lengths * torch.cos(thetas)
    # Vector from (x0, y0) to centroid
    vec_p0_c = torch.stack([cx - x0s, cy - y0s, cz - z0s], dim=1)
    # Direction vector of the segments
    vec_p0_p1 = torch.stack([x1s - x0s, y1s - y0s, z1s - z0s], dim=1)
    # Projection scalar of vec_p0_c onto vec_p0_p1
    dot_products = torch.sum(vec_p0_c * vec_p0_p1, dim=1)
    segment_lengths_squared = torch.sum(vec_p0_p1 * vec_p0_p1, dim=1)
    projection_scalars = dot_products / segment_lengths_squared
    # Clamp the projection_scalars to lie within the segment
    projection_scalars = torch.clamp(projection_scalars, min=0, max=1)
    # Calculate the nearest points on the segments to the centroid
    nearest_xs = x0s + projection_scalars * (x1s - x0s)
    nearest_ys = y0s + projection_scalars * (y1s - y0s)
    nearest_zs = z0s + projection_scalars * (z1s - z0s)
    # Distance from nearest points on the segments to the centroid
    distances = torch.sqrt(
        (nearest_xs - cx) ** 2 + (nearest_ys - cy) ** 2 + (nearest_zs - cz) ** 2
    )

    return distances


def get_dist_matrix_ls(data, centroids, dist_function):
    # init.
    dist_matrix = torch.zeros(data.shape[0], centroids.shape[0])
    for i in range(centroids.shape[0]):
        dist_matrix[:, i] = dist_function(data, centroids[i])
    return dist_matrix

def point_to_edge_distance(point, v1, v2):
    line_vec = v2 - v1
    point_vec = point - v1
    line_len = torch.sum(line_vec**2)
    if line_len == 0:
        return torch.sqrt(torch.sum(point_vec**2))
    t = torch.clamp(torch.dot(point_vec, line_vec) / line_len, 0, 1)
    projection = v1 + t * line_vec
    return torch.sqrt(torch.sum((point - projection) ** 2))


def point_to_polygon_distance(polygon, point):
    # Remove any infinity values to handle the case of non-convex polygon with fewer than 10 vertices
    valid_vertices = polygon[~torch.isinf(polygon).any(dim=1)]
    num_vertices = valid_vertices.size(0)

    min_distance = torch.tensor(float("inf"))

    for i in range(num_vertices):
        v1 = valid_vertices[i]
        v2 = valid_vertices[(i + 1) % num_vertices]

        distance = point_to_edge_distance(point, v1, v2)
        if distance < min_distance:
            min_distance = distance

    return min_distance

def get_dist_matrix_pls(data, centroids, dist_function):
    dist_matrix = torch.zeros(data.shape[0], centroids.shape[0])
    for i in range(centroids.shape[0]):
        dist_matrix[:, i] = torch.tensor([dist_function(poly, centroids[i]) for poly in data])
    return dist_matrix
