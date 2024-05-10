import torch

def compute_distances_2d(segments, centroid):
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


def get_dist_matrix(data, centroids, dist_function):
    # init.
    dist_matrix = torch.zeros(data.shape[0], centroids.shape[0])
    for i in range(centroids.shape[0]):
        dist_matrix[:, i] = dist_function(data, centroids[i])
    return dist_matrix
