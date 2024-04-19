import math
import warnings
import networkx as nx

from typing import Optional, Tuple, List, Dict
from pathlib import Path

from Node import Node
from utils.parse_graph_file import parse_graph, save_graph
from utils.constants import Directions as Dirs


class Graph:
    """
    A graph class with basic non-oriented graph operations.
    X-axis is the vertical axis, Y-axis is the horizontal axis, with (0,0) in the top left corner.
    """
    def __init__(self, file_path: Optional[str] = None):
        self.start = None
        self.objective = None
        self.graph = []
        self.shape = (0, 0)
        if file_path:
            self.start, self.objective, self.graph = parse_graph(Path(file_path))
            self.shape = self.get_shape()

    def set_start(self, node: Node):
        self.start = node

    def set_objective(self, node: Node):
        self.objective = node

    # Graph operations and methods
    def add_node(self, position: Tuple[int, int]):
        # The Graph needs to be extended if the position is out of bounds
        if position[0] < 0 or position[0] >= len(self.graph) or position[1] < 0 or position[1] >= len(self.graph[0]):
            self.graph = self.extend_graph(position)
            self.shape = self.get_shape()
        # The Node can be added if the position is empty
        if self.graph[position[0]][position[1]] is None:
            neighbors = self.get_neighbors(position)
            self.graph[position[0]][position[1]] = Node(position, neighbors)
            self.shape = self.get_shape()

    def add_edge(self, node1: Node, node2: Node, cost: int):
        if node1 not in self.graph:
            self.add_node(node1.position)
        if node2 not in self.graph:
            self.add_node(node2.position)
        if abs(node1.position[0] - node2.position[0]) + abs(node1.position[1] - node2.position[1]) <= math.sqrt(2):
            node1.add_neighbor(node2, cost)
            node2.add_neighbor(node1, cost)
        else:
            warnings.warn(f"Nodes {node1.position} and {node2.position} are not neighbors.")

    def remove_node(self, position: Tuple[int, int]):
        self.graph[position[0]][position[1]] = None
        self.clean()

    # Utils
    def extend_graph(self, position: Tuple[int, int]) -> List[List[Node]]:
        """Shift or add columns and lines to the graph to include the position."""
        if position[0] < 0:  # shift down
            self.graph.insert(0, [None for _ in range(len(self.graph[0]))])
        if position[1] < 0:  # shift right
            for row in self.graph:
                row.insert(0, None)
        if position[0] >= len(self.graph):  # add row
            self.graph.append([None for _ in range(len(self.graph[0]))])
        if position[1] >= len(self.graph[0]):  # add column
            for row in self.graph:
                row.append(None)
        return self.graph

    def clean(self):
        """Remove empty rows and columns from the graph."""
        self.graph = [row for row in self.graph if any(node is not None for node in row)]
        self.shape = self.get_shape()

    def get_neighbors(self, position: Tuple[int, int]) -> Dict[Tuple[int, int], Tuple[int, Node]]:
        """Return a neighbors for a given node."""
        x, y = position
        neighbors = {}
        maze_height, maze_width = self.shape
        for dx, dy in [Dirs.N, Dirs.S, Dirs.E, Dirs.W, Dirs.NE, Dirs.SE, Dirs.SW, Dirs.NW]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < maze_height and 0 <= new_y < maze_width:  # verify that we are inbounds
                neighbors[(dx, dy)] = (1, self.graph[new_x][new_y])
        return neighbors

    def convert_node_graph_to_nx(self) -> nx.Graph:
        """
        Converts a node graph into a weighted graph (for visualization)
        :return: weighted graph (nx graph)
        """

        nx_G = nx.Graph()

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                neighbors = self.get_neighbors((i, j))
                for neighbor in neighbors:
                    # TODO: Add edge to nx_G
                    pass
        return nx_G

    def save(self, file_path: str):
        save_graph(self.graph, Path(file_path))

    def get_shape(self):
        return len(self.graph), len(self.graph[0])

    def __getitem__(self, item):
        return self.graph[item]

    def __setitem__(self, key, value):
        self.graph[key] = value

    def __len__(self):
        return len(self.graph)

    def __iter__(self):
        return iter(self.graph)
