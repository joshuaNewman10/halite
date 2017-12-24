import math


def distance2(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def distance(x1, y1, x2, y2):
    return math.sqrt(distance2(x1, y1, x2, y2))


def angle(x, y):
    radians = math.atan2(y, x)
    if radians < 0:
        radians = radians + 2 * math.pi
    return round(radians / math.pi * 180)


def angle_dist(a1, a2):
    return (a1 - a2 + 360) % 360