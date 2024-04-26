def get_direction(direction: tuple[int, int]) -> str:
    if direction == Directions.N:
        return "N"
    elif direction == Directions.S:
        return "S"
    elif direction == Directions.E:
        return "E"
    elif direction == Directions.W:
        return "W"
    elif direction == Directions.NE:
        return "NE"
    elif direction == Directions.NW:
        return "NW"
    elif direction == Directions.SE:
        return "SE"
    elif direction == Directions.SW:
        return "SW"
    else:
        return "UNKNOWN"


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


class Colors:
    WHITE_COLOR = (255, 255, 255)
    BLACK_COLOR = (0, 0, 0)
    START_COLOR = (2, 159, 0)
    END_COLOR = (74, 120, 255)
    PATH_COLOR = (241, 138, 53)
    NX_COLOR = (31, 120, 180)

    def __init__(self):
        pass


class Strategies:
    A_STAR = "astar"
    CPLEX = "cplex"
