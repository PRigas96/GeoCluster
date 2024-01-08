import math
import random
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool

import numpy as np
import torch
from matplotlib import pyplot as plt
from math import pi, cos, sin, inf, ceil


def distance_between_points(p1: [int], p2: [int]):
    return math.sqrt(pow(p2[0] - p1[0], 2) + pow(p2[1] - p1[1], 2))


# An ellipse has an equation (x-h)^2/a^2 + (y-k)^2/b^2 = 1
class Ellipse:
    def __init__(self, h: float, a: float, k: float, b: float):
        self.h = h
        self.a = a
        self.k = k
        self.b = b

    def to_vector(self):
        return np.array([
            self.a,
            self.b,
            self.h,
            self.k
        ])


class Box:
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        self.ellipse = None
        self.x_min = x_min
        self.y_min = y_min
        self.y_max = y_max
        self.x_max = x_max
        self.draw_ellipse()
        # print(self.x_min, self.x_max, self.y_min, self.y_max)

    def draw_ellipse(self):
        h = (self.x_max + self.x_min) / 2
        k = (self.y_max + self.y_min) / 2
        a = self.x_max - self.x_min
        b = self.y_max - self.y_min
        self.ellipse = Ellipse(h, a, k, b)

    def is_inside(self, x: float, y: float):
        return self.x_min < x < self.x_max and self.y_min < y < self.y_max

    def distance(self, x: float, y: float):
        if self.x_min < x < self.x_max:
            x_dist = -1
        else:
            x_dist = min(abs(x - self.x_min), abs(x - self.x_max))
        if self.y_min < y < self.y_max:
            y_dist = -1
        else:
            y_dist = min(abs(y - self.y_min), abs(y - self.y_max))
        # print(x, y)
        # print(self.x_min, self.x_max, self.y_min, self.y_max)
        # print(x_dist, y_dist)
        return max(x_dist, y_dist)


def get_distance(box, center):
    # dist = inf
    # for box in boxes:
    #     dist = min(dist, box.distance(center[0], center[1]))
    return box.distance(center[0], center[1])


def get_distance_(data):
    # dist = inf
    # for box in boxes:
    #     dist = min(dist, box.distance(center[0], center[1]))
    return data[0].distance(data[1][0], data[1][1])


def get_boxes(num_of_ellipses: int, x_from: float, x_to: float, y_from: float, y_to: float,
              length_min: float, length_max: float, height_min: float, height_max: float):
    boxes = []
    for i in range(num_of_ellipses):
        if i % 100 == 0:
            print(i)
        while True:
            center = [random.uniform(x_from + length_max / 2, x_to - length_max / 2),
                      random.uniform(y_from + height_max / 2, y_to - height_max / 2)]
            stop = True
            for box in boxes:
                if box.is_inside(center[0], center[1]):
                    # print("aaaaaa")
                    stop = False
                    break
            if stop:
                break
        size = [random.uniform(length_min, length_max), random.uniform(height_min, height_max)]
        distance = inf
        for box in boxes:
            distance = min(distance, box.distance(center[0], center[1]))
        # print("s", size)
        size = [min(size[0], distance), min(size[1], distance)]
        # print(result, distance)
        boxes.append(
            Box(center[0] - size[0] / 2, center[0] + size[0] / 2, center[1] - size[1] / 2, center[1] + size[1] / 2))
    return boxes


def draw_ellipses(boxes: [Box]):
    t = np.linspace(0, 2 * pi, 100)
    for box in boxes:
        plt.plot(box.ellipse.h + box.ellipse.a * np.cos(t), box.ellipse.k + box.ellipse.b * np.sin(t))
    plt.grid(color='lightgray', linestyle='--')
    plt.show()


def plot_edge(e, color):
    x = np.linspace(e[0][0], e[1][0], 100)
    y = np.linspace(e[0][1], e[1][1], 100)
    plt.plot(x, y, color)


def plot_square(s, color):
    vertices = [
        [s.x_min, s.y_min],
        [s.x_max, s.y_min],
        [s.x_max, s.y_max],
        [s.x_min, s.y_max]
    ]
    square_edges = [
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[3]],
        [vertices[3], vertices[0]],
    ]
    for se in square_edges:
        plot_edge(se, color)


def save_to_file(file_name: str, boxes: [Box]):
    objs = []
    for box in boxes:
        objs.append(np.array([
            box.ellipse.a,
            box.ellipse.b,
            box.ellipse.h,
            box.ellipse.k
        ]))
    np.save(file_name, np.array(objs, dtype=object), allow_pickle=True)
