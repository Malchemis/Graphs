import math
import warnings
import networkx as nx
import numpy as np
import cv2
import matplotlib.pyplot as plt

from typing import Optional, Tuple, List, Dict
from pathlib import Path

from Node import Node
from utils.parse_graph_file import parse_graph, save_graph
from utils.constants import Directions as Dirs, Colors, Strategies
from search import aStar


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
        self.file_path = file_path
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

    def solve(self, strategy: Strategies = Strategies.A_STAR) -> Optional[List[Node]]:
        if self.start is None or self.objective is None:
            warnings.warn("Start or objective node not set.")
            return
        if strategy == Strategies.A_STAR:
            return aStar(self.start, self.objective)
        else:
            warnings.warn(f"Strategy {strategy} not implemented.")
            return

    def solve_and_save(self, strategy: Strategies = Strategies.A_STAR):
        if self.file_path is None:
            raise ValueError("Couldn't save the solution: File path not set.")
        path = self.solve(strategy)

        if path is None :
            print("Couldn't save the solution: No path found.")
            return

        for node in path:
            pass
            #TODO: Save the path in file


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
            if 0 <= new_x < maze_height and 0 <= new_y < maze_width and self.graph[new_x][new_y] is not None:
                neighbors[(dx, dy)] = (1, self.graph[new_x][new_y])
        return neighbors

    def display_network(self, path: Optional[List[Node]] = None):
        """
        Display the graph as a networkx graph.
        :return: weighted graph (nx graph)
        """

        # Convert the graph to a networkx graph
        nx_G = nx.Graph()

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.graph[i][j] is not None:
                    neighbors = self.get_neighbors((i, j))
                    node_id = i * self.shape[1] + j
                    for neighbor in neighbors:
                        neighbor_id = (i + neighbor[0]) * self.shape[1] + j + neighbor[1]
                        # Add nodes
                        if not nx_G.has_edge(neighbor_id, node_id):
                            # Find color
                            color = "#1f78b4"
                            color_neighbor = "#1f78b4"
                            edge_color = "black"
                            if path is not None:
                                # Path color
                                if self.graph[i][j] in path:
                                    if self.graph[i][j] == self.start:
                                        color = "#029f00"
                                    elif self.graph[i][j] == self.objective:
                                        color = "#4a78ff"
                                    else:
                                        color = "red"
                                if self.graph[i + neighbor[0]][j + neighbor[1]] in path:
                                    if self.graph[i + neighbor[0]][j + neighbor[1]] == self.start:
                                        color_neighbor = "#029f00"
                                    elif self.graph[i + neighbor[0]][j + neighbor[1]] == self.objective:
                                        color_neighbor = "#4a78ff"
                                    else:
                                        color_neighbor = "red"
                                if color != "#1f78b4" and color_neighbor != "#1f78b4":
                                    edge_color = "red"

                            nx_G.add_node(node_id, pos=(j,len(self.graph)-i), color=color)
                            nx_G.add_node(neighbor_id, pos=(j + neighbor[1], len(self.graph) - (i + neighbor[0])), color=color_neighbor)
                            nx_G.add_edge(node_id, neighbor_id, weight=neighbors[neighbor][0], color=edge_color)

        # Display the graph
        fig = plt.figure(figsize=(len(self.graph[0])/2, len(self.graph)/2))
        node_color_map = nx.get_node_attributes(nx_G, 'color').values()
        edge_color_map = nx.get_edge_attributes(nx_G, 'color').values()
        pos = nx.get_node_attributes(nx_G, 'pos')
        nx.draw_networkx(nx_G, with_labels=False, pos=pos, node_color=node_color_map, edge_color=edge_color_map)
        plt.show()

    def display(self, path: Optional[List[Node]] = None):
        """Display the graph in a window. Press 'q' to close."""

        # Setup
        img = np.zeros((len(self.graph), len(self.graph[0]), 3), dtype=np.uint8)

        # Display
        for i in range(len(img)):
            for j in range(len(img[0])):
                if self.graph[i][j] is None:
                    img[i][j] = Colors.BLACK_COLOR
                elif self.graph[i][j] == self.start:
                    img[i][j] = Colors.START_COLOR
                elif self.graph[i][j] == self.objective:
                    img[i][j] = Colors.END_COLOR
                else:
                    if path is not None:
                        if self.graph[i][j] in path:
                            img[i][j] = Colors.PATH_COLOR
                        else:
                            img[i][j] = Colors.WHITE_COLOR
                    else:
                        img[i][j] = Colors.WHITE_COLOR

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (len(self.graph[0]) * 50, len(self.graph) * 50), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Graph", img)
        key = cv2.waitKey(10)

        # Wait for 'q' key to close
        while key != ord("q"):
            key = cv2.waitKey(10)
        cv2.destroyAllWindows()
        return

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

    def __str__(self):
        string_graph = "Start: " + str(self.start) + "\nObjective: " + str(self.objective) + "\n"
        for i in range(len(self.graph)):
            for j in range(len(self.graph[0])):
                string_graph += f"{self.graph[i][j]} \n"
        return string_graph
