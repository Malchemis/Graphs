"""
A* Pathfinding algorithm implementation
"""

from typing import List, Optional
import heapq
from Node import Node


def a_star(start_node: Node, end_node: Node) -> Optional[List[Node]]:
    """
    A* Pathfinding algorithm implementation. Minimizes distance and changes of orientation
    :param start_node: the starting node / points
    :param end_node: the objective node / points
    :return: a list of nodes from start to end, or None if no path is found
    """
    open_set = [start_node]
    heapq.heapify(open_set)
    closed_set = set()

    start_node.g = 0
    start_node.h = heuristic(start_node, end_node)
    start_node.f = start_node.h

    # Loop until the open list is empty
    while open_set:
        # Use a priority queue to get the node with the lowest f value
        current_node = heapq.heappop(open_set)  # Also remove from open set
        closed_set.add(current_node)

        # Path found
        if current_node == end_node:
            return reconstruct_path(current_node)

        # Explore neighbors
        for _, neighbor in current_node.neighbors.values():
            if neighbor is None:
                continue

            tentative_g = current_node.g + euclidean(current_node, neighbor)  # Distance from start to neighbor
            if tentative_g < neighbor.g:
                neighbor.parent = current_node
                neighbor.g = tentative_g
                neighbor.h = heuristic(neighbor, end_node)
                neighbor.f = neighbor.g + neighbor.h
                if neighbor not in closed_set:  # To be visited
                    heapq.heappush(open_set, neighbor)

    print("(A*) No path found")
    return None


def reconstruct_path(node):
    path = []
    # Reconstruct path from end to start
    while node:
        path.append(node)
        node = node.parent
    # Reverse path
    return path[::-1]


def euclidean(node1: Node, node2: Node) -> float:
    """
    Euclidean distance between two nodes
    :param node1: node 1
    :param node2: node 2
    :return: euclidean distance between the two nodes
    """
    return ((node1.position[0] - node2.position[0]) ** 2 + (node1.position[1] - node2.position[1]) ** 2) ** 0.5


def heuristic(node: Node, end_node: Node) -> float:
    """
    Heuristic function
    :param node: current node
    :param end_node: objective node
    :return: heuristic value
    """
    # We could use the Manhattan distance instead. For our case, the Euclidian distance is better
    return euclidean(node, end_node)
