import math
import random
import matplotlib.pyplot as plt
import numpy as np


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


def sample_points_sphere(points, center, radius):
    """
    Sample the points within a sphere, specified by the given center and redius
    """
    distance = np.sqrt(calc_distances(center, points))
    sampled_points = points[distance <= radius, :]

    return sampled_points


def test_fps():
    num_samples = 100

    # make a simple unit circle
    theta = np.linspace(0, 2 * np.pi, num_samples)
    a, b = 1 * np.cos(theta), 1 * np.sin(theta)

    r = 1
    x, y = r * np.cos(theta), r * np.sin(theta)
    plt.figure(figsize=(7, 6))
    plt.plot(a, b, linewidth=2, label='Circle')
    plt.plot(x, y, marker='o', label='Samples')
    plt.ylim([-1.5, 1.5])
    plt.xlim([-1.5, 1.5])
    plt.grid()
    plt.legend(loc='upper right')

    points = np.array([x, y])
    points = np.transpose(points)
    res = farthest_points_sampling(points, 9)
    plt.plot(res[:, 0], res[:, 1], marker='o', color='r', label='Samples')
    plt.axis('equal')

    res = sample_points_sphere(points, points[0], 1)
    plt.plot(res[:, 0], res[:, 1], marker='o', color='b', label='Samples')
    plt.axis('equal')

    plt.show(block=True)
