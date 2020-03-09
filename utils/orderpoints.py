import numpy as np


def order_points(pts, plane_num=32, range=(-1.0, 1.0)):
    order_pts_x_dir, quantiles_x_dir = order_points_x_axis(pts, plane_num, range)
    order_pts_y_dir, quantiles_y_dir = order_points_y_axis(pts, plane_num, range)
    order_pts_z_dir, quantiles_z_dir = order_points_z_axis(pts, plane_num, range)

    ordered_pts = np.stack((order_pts_x_dir, order_pts_y_dir, order_pts_z_dir), axis=0)
    quantiles = np.stack((quantiles_x_dir, quantiles_y_dir, quantiles_z_dir), axis=0)

    return ordered_pts, quantiles


def order_points_y_axis(pts, y_plane_num=32, y_range=(-1.0, 1.0)):
    """
    pts: Nx3 point sets
    """
    ordered_pts = []
    quantiles = []
    y_plane_base = y_range[0]
    delta = (y_range[1] - y_range[0]) / y_plane_num

    for i in range(y_plane_num):
        y_plane = y_plane_base + i * delta
        center_xyz = np.array([[0.0, y_plane, 0.0]])

        if i == 0:
            band_pts_idx = np.where(pts[:, 1] <= y_plane + delta)[0]
        elif i == y_plane_num - 1:
            band_pts_idx = np.where(y_plane < pts[:, 1])[0]
        else:
            band_pts_idx = np.where((y_plane < pts[:, 1]) & (pts[:, 1] <= y_plane + delta))[0]
        band_pts = pts[band_pts_idx, :]

        num_band_pts = band_pts_idx.size
        quantiles.append(num_band_pts)

        # split points into two halves
        # x positive parts
        band_pts_x_pos_idx = np.where(band_pts[:, 0] >= 0)[0]
        band_pts_x_pos = band_pts[band_pts_x_pos_idx, ]
        band_pts_xz_plane = band_pts_x_pos[:, (0, 2)]

        distance_xyz = np.linalg.norm(band_pts_x_pos - center_xyz, 2, axis=1)
        distance_xz_plane = np.linalg.norm(band_pts_xz_plane, 2, axis=1)
        sin_angle = band_pts_x_pos[:, 2] / distance_xz_plane
        pitch_angle = (band_pts_x_pos[:, 1] - y_plane) / distance_xyz

        ind_x_pos = np.lexsort((pitch_angle, distance_xyz, sin_angle))
        ordered_pts.append(band_pts_x_pos[ind_x_pos, ])

        # x negative parts
        band_pts_x_neg_idx = np.where(band_pts[:, 0] < 0)[0]
        band_pts_x_neg = band_pts[band_pts_x_neg_idx, ]
        band_pts_xz_plane = band_pts_x_neg[:, (0, 2)]

        distance_xyz = np.linalg.norm(band_pts_x_neg - center_xyz, 2, axis=1)
        distance_xz_plane = np.linalg.norm(band_pts_xz_plane, 2, axis=1)
        sin_angle = band_pts_x_neg[:, 2] / distance_xz_plane
        pitch_angle = (band_pts_x_neg[:, 1] - y_plane) / distance_xyz

        ind_x_neg = np.lexsort((pitch_angle, distance_xyz, sin_angle))[::-1]
        ordered_pts.append(band_pts_x_neg[ind_x_neg, ])

    ordered_pts = np.concatenate(ordered_pts)

    return ordered_pts, quantiles


def order_points_x_axis(pts, x_plane_num=32, x_range=(-1.0, 1.0)):
    """
    pts: Nx3 point sets
    """
    ordered_pts = []
    quantiles = []
    x_plane_base = x_range[0]
    delta = (x_range[1] - x_range[0]) / x_plane_num

    for i in range(x_plane_num):
        x_plane = x_plane_base + i * delta
        center_xyz = np.array([[x_plane, 0.0, 0.0]])

        if i == 0:
            band_pts_idx = np.where(pts[:, 0] <= x_plane + delta)[0]
        elif i == x_plane_num - 1:
            band_pts_idx = np.where(x_plane < pts[:, 0])[0]
        else:
            band_pts_idx = np.where((x_plane < pts[:, 0]) & (pts[:, 0] <= x_plane + delta))[0]
        band_pts = pts[band_pts_idx, :]

        num_band_pts = band_pts_idx.size
        quantiles.append(num_band_pts)

        # split points into two halves
        # z positive parts
        band_pts_z_pos_idx = np.where(band_pts[:, 2] >= 0)[0]
        band_pts_z_pos = band_pts[band_pts_z_pos_idx, ]
        band_pts_yz_plane = band_pts_z_pos[:, (1, 2)]

        distance_xyz = np.linalg.norm(band_pts_z_pos - center_xyz, 2, axis=1)
        distance_yz_plane = np.linalg.norm(band_pts_yz_plane, 2, axis=1)
        sin_angle = band_pts_z_pos[:, 1] / distance_yz_plane
        pitch_angle = (band_pts_z_pos[:, 0] - x_plane) / distance_xyz

        ind_z_pos = np.lexsort((pitch_angle, distance_xyz, sin_angle))
        ordered_pts.append(band_pts_z_pos[ind_z_pos, ])

        # z negative parts
        band_pts_z_neg_idx = np.where(band_pts[:, 2] < 0)[0]
        band_pts_z_neg = band_pts[band_pts_z_neg_idx, ]
        band_pts_yz_plane = band_pts_z_neg[:, (1, 2)]

        distance_xyz = np.linalg.norm(band_pts_z_neg - center_xyz, 2, axis=1)
        distance_yz_plane = np.linalg.norm(band_pts_yz_plane, 2, axis=1)
        sin_angle = band_pts_z_neg[:, 1] / distance_yz_plane
        pitch_angle = (band_pts_z_neg[:, 0] - x_plane) / distance_xyz

        ind_z_neg = np.lexsort((pitch_angle, distance_xyz, sin_angle))[::-1]
        ordered_pts.append(band_pts_z_neg[ind_z_neg, ])

    ordered_pts = np.concatenate(ordered_pts)

    return ordered_pts, quantiles


def order_points_z_axis(pts, z_plane_num=32, z_range=(-1.0, 1.0)):
    """
    pts: Nx3 point sets
    """
    ordered_pts = []
    quantiles = []
    z_plane_base = z_range[0]
    delta = (z_range[1] - z_range[0]) / z_plane_num

    for i in range(z_plane_num):
        z_plane = z_plane_base + i * delta
        center_xyz = np.array([[0.0, 0.0, z_plane]])

        if i == 0:
            band_pts_idx = np.where(pts[:, 2] <= z_plane + delta)[0]
        elif i == z_plane_num - 1:
            band_pts_idx = np.where(z_plane < pts[:, 2])[0]
        else:
            band_pts_idx = np.where((z_plane < pts[:, 2]) & (pts[:, 2] <= z_plane + delta))[0]
        band_pts = pts[band_pts_idx, :]

        num_band_pts = band_pts_idx.size
        quantiles.append(num_band_pts)

        # split points into two halves
        # y positive parts
        band_pts_y_pos_idx = np.where(band_pts[:, 1] >= 0)[0]
        band_pts_y_pos = band_pts[band_pts_y_pos_idx, ]
        band_pts_xy_plane = band_pts_y_pos[:, (0, 1)]

        distance_xyz = np.linalg.norm(band_pts_y_pos - center_xyz, 2, axis=1)
        distance_xy_plane = np.linalg.norm(band_pts_xy_plane, 2, axis=1)
        sin_angle = band_pts_y_pos[:, 0] / distance_xy_plane
        pitch_angle = (band_pts_y_pos[:, 2] - z_plane) / distance_xyz

        ind_y_pos = np.lexsort((pitch_angle, distance_xyz, sin_angle))
        ordered_pts.append(band_pts_y_pos[ind_y_pos, ])

        # y negative parts
        band_pts_y_neg_idx = np.where(band_pts[:, 1] < 0)[0]
        band_pts_y_neg = band_pts[band_pts_y_neg_idx, ]
        band_pts_xy_plane = band_pts_y_neg[:, (0, 1)]

        distance_xyz = np.linalg.norm(band_pts_y_neg - center_xyz, 2, axis=1)
        distance_xy_plane = np.linalg.norm(band_pts_xy_plane, 2, axis=1)
        sin_angle = band_pts_y_neg[:, 0] / distance_xy_plane
        pitch_angle = (band_pts_y_neg[:, 2] - z_plane) / distance_xyz

        ind_y_neg = np.lexsort((pitch_angle, distance_xyz, sin_angle))[::-1]
        ordered_pts.append(band_pts_y_neg[ind_y_neg, ])

    ordered_pts = np.concatenate(ordered_pts)

    return ordered_pts, quantiles


def points_grid(pts, plane_num=16, extension=(-1.0, 1.0)):
    plane_base = extension[0]
    vline_base = extension[0]
    hline_base = extension[0]

    grid_resolution = plane_num

    delta = (extension[1] - extension[0]) / plane_num
    grid_delta = (extension[1] - extension[0]) / grid_resolution

    quantiles = []
    ordered_pts = []

    for i in range(plane_num):
        plane = plane_base + i * delta

        if i == 0:
            upper_bound = plane + delta
            lower_bound = -np.inf
        elif i == plane_num - 1:
            upper_bound = np.inf
            lower_bound = plane
        else:
            upper_bound = plane + delta
            lower_bound = plane

        quant = []
        for j in range(grid_resolution):
            hline = hline_base + j * grid_delta

            if j == 0:
                h_upper_bound = hline + grid_delta
                h_lower_bound = -np.inf
            elif j == grid_resolution - 1:
                h_upper_bound = np.inf
                h_lower_bound = hline
            else:
                h_upper_bound = hline + grid_delta
                h_lower_bound = hline

            grid_cell_idx = np.where((h_lower_bound < pts[:, 0]) & (pts[:, 0] <= h_upper_bound) &
                                     (lower_bound < pts[:, 1]) & (pts[:, 1] <= upper_bound))[0]
            grid_cell = pts[grid_cell_idx, :]

            quant.append(grid_cell_idx.size)

            sort_idx = np.lexsort((grid_cell[:, 2], grid_cell[:, 1], grid_cell[:, 0]))
            ordered_pts.append(grid_cell[sort_idx, ])

        quantiles.append(quant)

    ordered_pts = np.concatenate(ordered_pts)
    quantiles = np.array(quantiles)

    return ordered_pts, quantiles


def farthest_points_sampling(points, k):
    """
    Perform the farthest points sampling.
    Input:
        points: a point set, in the format of NxM, where N is the number of points, and M is the point dimension
        k: required number of sampled points
    """
    points = np.array(points)  # make sure it is a numpy array
    farthest_pts = np.zeros((k, points.shape[1]))

    farthest_pts[0] = points[np.random.randint(len(points))]
    distances = calc_distances(farthest_pts[0], points)

    for i in range(1, k):
        farthest_pts[i] = points[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], points))

    return farthest_pts


def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


if __name__ == '__main__':
    # pts = np.array([[1, 0, 0], [0, 0, 1], [0., 0.0, -1.], [-1, 0, 0]])
    pts = np.random.rand(1000, 3)
    # print(pts)
    res, q = points_grid(pts)
    print(res, q)
