"""
A* Pathfinding algorithm implementation
"""

from typing import List, Optional
import heapq
from Node import Node


def aStar(start_node: Node, end_node: Node) -> Optional[List[Node]]:
    """
    A* Pathfinding algorithm implementation
    :param start_node: the starting node / points
    :param end_node: the objective node / points
    :return: a list of nodes from start to end, or None if no path is found
    """
    open_list = [start_node]
    closed_list = set()
    previous_node = start_node

    # Loop until the open list is empty
    while open_list:
        # Use a priority queue to get the node with the lowest f value
        current_node = heapq.heappop(open_list)
        if previous_node != current_node:   # Update the parent node
            current_node.parent = previous_node
            previous_node = current_node

        if current_node == end_node:  # Path found
            path = []
            current = current_node
            while current is not None:
                path.append(current)
                current = current.parent
            return path[::-1]

        closed_list.add(current_node)   # Mark as visited

        # Explore neighbors
        for cost, neighbor in current_node.neighbors.values():
            if neighbor is None:  # WALL / OBSTACLE
                continue

            # Compute the new cost
            neighbor.g = current_node.g + euclidian(neighbor, current_node)  # update g value
            neighbor.h = heuristic(neighbor, end_node)
            neighbor.f = cost * neighbor.g + neighbor.h  # f = alpha * g + beta * h. Here alpha = cost and beta = 1

            if neighbor in closed_list:  # Skip if already visited
                continue

            if neighbor not in open_list:  # To be visited
                heapq.heappush(open_list, neighbor)

    return None


def euclidian(node1: Node, node2: Node) -> float:
    """
    Euclidian distance between two nodes
    :param node1: node 1
    :param node2: node 2
    :return: euclidian distance between the two nodes
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
    return euclidian(node, end_node)
