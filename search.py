"""
A* Pathfinding algorithm implementation
"""

from typing import List, Optional
import heapq
from Node import Node
from utils.distances import manhattan, euclidean
import mip


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
    model = mip.Model()
    # Create variables for each potential edge in the graph
    edge_vars = {}
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            node = graph[i][j]
            if node.is_obstacle:
                continue
            for direction, (cost, neighbor) in node.neighbors.items():
                if (node, neighbor) not in edge_vars and (neighbor, node) not in edge_vars:
                    edge_vars[(node, neighbor)] = model.add_var(var_type=mip.BINARY)

    # Objective function: Minimize total cost of the path
    model.objective = mip.xsum(edge_vars[edge] * edge[0].neighbors[(edge[1].position[0] - edge[0].position[0], edge[1].position[1] - edge[0].position[1])][0] for edge in edge_vars)

    # Constraints
    # Each node (except start and end) should have exactly two adjacent edges selected
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            node = graph[i][j]
            if node.is_obstacle or node == graph.start or node == graph.objective:
                continue
            model += (mip.xsum(edge_vars[(node, neighbor)] for direction, (cost, neighbor) in node.neighbors.items() if (node, neighbor) in edge_vars or (neighbor, node) in edge_vars) == 2)

    # Start and end nodes
    if graph.start:
        model += (mip.xsum(edge_vars[(graph.start, neighbor)] for direction, (cost, neighbor) in graph.start.neighbors.items() if (graph.start, neighbor) in edge_vars) == 1)
    if graph.objective:
        model += (mip.xsum(edge_vars[(graph.objective, neighbor)] for direction, (cost, neighbor) in graph.objective.neighbors.items() if (graph.objective, neighbor) in edge_vars) == 1)

    # Solve the model
    model.optimize()

    # Retrieve the solution
    if model.num_solutions:
        path_edges = [edge for edge in edge_vars if edge_vars[edge].x >= 0.99]
        return path_edges
    else:
        return None
