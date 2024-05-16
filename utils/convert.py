import numpy as np
from typing import List

from Node import Node
from utils.constants import Values


def convert_to_node_graph(graph: np.ndarray):
    """
    Converts a numpy array into a node graph.
    :param graph: np matrix
    :return: start node, end node, node graph
    """
    node_graph: List[List[Node | None]] = [[None for _ in range(graph.shape[1])] for _ in range(graph.shape[0])]
    start_node = None
    end_node = None
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            current_is_obstacle = graph[i][j] == Values.WALL
            node_graph[i][j] = Node(position=(i, j), neighbors={}, is_obstacle=current_is_obstacle)
            if i - 1 >= 0 and j - 1 >= 0 and graph[i - 1][j - 1] != 0:
                node_graph[i][j].add_neighbor(node_graph[i - 1][j - 1], 1)
            if i - 1 >= 0 and graph[i - 1][j] != 0:
                node_graph[i][j].add_neighbor(node_graph[i - 1][j], 1)
            if j - 1 >= 0 and graph[i][j - 1] != 0:
                node_graph[i][j].add_neighbor(node_graph[i][j - 1], 1)
            if i - 1 >= 0 and j + 1 < graph.shape[1] and graph[i - 1][j + 1] != 0:
                node_graph[i][j].add_neighbor(node_graph[i - 1][j + 1], 1)

            if graph[i][j] == Values.START:
                start_node = node_graph[i][j]
            elif graph[i][j] == Values.OBJECTIVE:
                end_node = node_graph[i][j]
    return start_node, end_node, node_graph

