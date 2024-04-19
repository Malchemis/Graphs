class Directions:
    """Directions go from top left to bottom right"""
    N = (-1, 0)
    S = (1, 0)
    E = (0, 1)
    W = (0, -1)
    NE = (-1, 1)
    NW = (-1, -1)
    SE = (1, 1)
    SW = (1, -1)

    def __init__(self):
        pass


class Values:
    WALL = 0
    START = 2
    OBJECTIVE = 3
    EMPTY = 1

    def __init__(self):
        pass