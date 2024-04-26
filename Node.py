from typing import Optional, Tuple, Dict, Any

from utils.constants import get_direction


class Node:
    """
    A node class for A* Pathfinding and Graph search
    """

    def __init__(self, position: Tuple[int, int],
                 neighbors: Optional[Dict[Tuple[int, int], Tuple[int, Any]]] = None,
                 parent: Optional['Node'] = None):
        """
        :param neighbors: dictionary with keys as directions and values as cost
        :param parent: the parent node. Optional, used for path reconstruction
        :param position: the (x, y) position of the node
        """
        self.parent = parent
        self.position = position
        self.neighbors = neighbors
        self.g = 0
        self.h = 0
        self.f = 0

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
        neighbors_directions = []
        try :
            for direction, neighbor in self.neighbors.items():
                neighbors_directions.append(f'{get_direction(direction)}: {neighbor[0]}')
        except:
            pass
        return f'Node at {self.position}, neighbors: {neighbors_directions}'
