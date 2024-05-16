from Node import Node


def manhattan(node1: Node, node2: Node) -> float:
    """
    Manhattan distance between two nodes
    :param node1: node 1
    :param node2: node 2
    :return: manhattan distance between the two nodes
    """
    return abs(node1.position[0] - node2.position[0]) + abs(node1.position[1] - node2.position[1])


def euclidean(node1: Node, node2: Node) -> float:
    """
    Euclidean distance between two nodes
    :param node1: node 1
    :param node2: node 2
    :return: euclidean distance between the two nodes
    """
    return ((node1.position[0] - node2.position[0]) ** 2 + (node1.position[1] - node2.position[1]) ** 2) ** 0.5