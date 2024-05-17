import numpy as np

from pathlib import Path
from typing import List

from Node import Node
from utils.constants import Values


def parse_graph(path: Path):
    with open(path) as f:
        lines = f.readlines()
        n, m = map(int, lines[0].split())
        graph = np.zeros((n, m))

        index = 0
        for line in lines[1:]:
            parsed_line = line.split(" ")
            if parsed_line[-1] == "\n":
                parsed_line.remove("\n")
            graph[index] = np.array(list(map(int, parsed_line)))
            index += 1
    return convert_to_node_graph(graph)


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


def parse_tsp(path: Path):
    """Returns list of nodes and dictionary of edges with their cost.
    file struct:
    <number of vertices> <number of edges>
    <node1> <node2> <cost>
    ...
    """
    with open(path, 'r') as file:
        lines = file.readlines()

    nodes = []
    cost = {}
    for line in lines[1:]:
        node1, node2, c = map(int, line.split())
        if node1 not in nodes:
            nodes.append(node1)
        if node2 not in nodes:
            nodes.append(node2)
        cost[(node1, node2)] = c
        cost[(node2, node1)] = c

    return nodes, cost
