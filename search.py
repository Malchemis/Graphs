"""
A* Pathfinding algorithm implementation
"""

from typing import List, Optional
import heapq
from cplex import Cplex
from docplex.mp.model import Model

from Node import Node
from utils.distances import manhattan, euclidean


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

            tentative_g = current_node.g + distance(current_node, neighbor)  # Distance from start to neighbor
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


def distance(node1: Node, node2: Node) -> float:
    """
    Distance between two nodes
    :param node1: node 1
    :param node2: node 2
    :return: distance between the two nodes
    """
    # Manhattan / Euclidean distance
    # return euclidean(node1, node2)
    return manhattan(node1, node2)


def heuristic(node: Node, end_node: Node) -> float:
    """
    Heuristic function
    :param node: current node
    :param end_node: objective node
    :return: heuristic value
    """
    # We could use the Manhattan distance instead. For our case, the Euclidian distance is better
    return euclidean(node, end_node)


def cplex_solver(graph, start_node, end_node):
    model = Model("Pathfinding")
    edges = graph.get_edges()

    # Variables
    x = model.binary_var_dict(edges, name="x")

    # Objective function
    model.minimize(model.sum(
        edge[2] * x[edge] for edge in edges
    ))

    # Constraints
    # Each node has exactly one outgoing edge
    for edge in edges:
        model.add_constraint(
            model.sum(x[(i, j, cost)] for i, j, cost in edges if i == edge[0]) == 1
        )

    # Each node has exactly one incoming edge
    for edge in edges:
        model.add_constraint(
            model.sum(x[(i, j, cost)] for i, j, cost in edges if j == edge[1]) == 1
        )

    # Start node has one outgoing edge
    for j, cost in graph.get_neighbors(start_node.position):
        print(f"Start node: {start_node.position} -> {j} with cost {cost}")
        model.add_constraint(x[(start_node, j, cost)] == 1)

    model.add_constraint(
        model.sum(x[(start_node, j, cost)] for j, cost in graph.get_neighbors(start_node.position)) == 1
    )

    # End node has one incoming edge
    model.add_constraint(
        model.sum(x[(i, end_node, cost)] for i, cost in graph.get_neighbors(end_node.position)) == 1
    )


    model.solve()
    print(model.solution)
