import numpy as np
from pathlib import Path

from Node import Node
from utils.constants import Values, Directions as Dirs


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
    node_graph = [[Node(position=(i, j), is_obstacle=graph[i][j] == Values.WALL) for j in range(graph.shape[1])]
                  for i in range(graph.shape[0])]
    start_node = None
    end_node = None

    # add neighbors to the graph
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if graph[i][j] == Values.START:
                start_node = node_graph[i][j]
            elif graph[i][j] == Values.OBJECTIVE:
                end_node = node_graph[i][j]
            for dx, dy in [Dirs.N, Dirs.S, Dirs.E, Dirs.W, Dirs.NE, Dirs.SE, Dirs.SW, Dirs.NW]:
                x, y = i + dx, j + dy
                if 0 <= x < graph.shape[0] and 0 <= y < graph.shape[1]:
                    node_graph[i][j].add_neighbor(node_graph[x][y])

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
