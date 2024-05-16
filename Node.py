from typing import Optional, Tuple, Dict, Any

class Node:
    """
    A node class for A* Pathfinding and CPLEX Graph search
    """
    def __init__(self, position: Tuple[int, int],
                 neighbors: Optional[Dict[Tuple[int, int], Tuple[int, Any]]] = None,
                 parent: Optional['Node'] = None, is_obstacle: bool = False):
        """
        :param neighbors: dictionary with keys as directions and values as cost
        :param parent: the parent node. Optional, used for path reconstruction
        :param position: the (x, y) position of the node
        """
        self.parent = parent
        self.position = position
        self.neighbors = neighbors
        self.g = float('inf')
        self.h = 0
        self.f = 0
        self.is_obstacle = is_obstacle

    def add_neighbor(self, node_neighbor: 'Node', cost: int):
        direction = (node_neighbor.position[0] - self.position[0], node_neighbor.position[1] - self.position[1])
        self.neighbors[direction] = (cost, node_neighbor)
        node_neighbor.neighbors[(-direction[0], -direction[1])] = (cost, self)

    def __eq__(self, other: 'Node') -> bool:
        return self.position == other.position

    def __lt__(self, other: 'Node') -> bool:
        return self.f < other.f

    def __hash__(self) -> int:
        return hash(self.position)

    def __str__(self) -> str:
        return f'({self.position[0]}, {self.position[1]})' if not self.is_obstacle else 'WALL'

    def __repr__(self) -> str:
        return self.__str__()
