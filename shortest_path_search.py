"""
Pathfinding algorithms :
- A* Pathfinding algorithm implementation
- CPLEX solver for pathfinding
"""

from typing import List, Optional
import heapq

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
    open_set_tracker = {start_node}  # used for faster membership checking

    start_node.g = 0
    start_node.h = heuristic(start_node, end_node)
    start_node.f = start_node.h

    # Loop until the open list is empty
    while open_set:
        # Use a priority queue to get the node with the lowest f value
        current_node: Node = heapq.heappop(open_set)  # Also remove from open set
        open_set_tracker.remove(current_node)

        # Path found
        if current_node == end_node:
            return reconstruct_path(current_node)

        # Explore neighbors
        for neighbor in current_node.neighbors.values():
            tentative_g = current_node.g + distance(current_node, neighbor)  # Distance from start to neighbor
            if tentative_g < neighbor.g:
                neighbor.parent = current_node
                neighbor.g = tentative_g
                neighbor.h = heuristic(neighbor, end_node)
                neighbor.f = neighbor.g + neighbor.h
                if neighbor not in open_set_tracker:  # To be visited
                    heapq.heappush(open_set, neighbor)
                    open_set_tracker.add(neighbor)

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
    # return manhattan(node, end_node)
    return euclidean(node, end_node)
    # return 0


def shortest_path_cplex_solver(graph, start_node, end_node):
    model = Model("Pathfinding")
    edges = graph.cost.keys()
    cost = graph.cost

    # Variables for each edge: 1 if edge is used, 0 otherwise
    x = {e: model.binary_var(name=f"x_{e[0].position}_{e[1].position}") for e in edges}

    # Objective: Minimize the sum of the costs of the edges included in the path
    model.minimize(model.sum(cost[e] * x[e] for e in edges))

    # Constraints
    # Ensure exactly one outgoing edge from start and one incoming edge to end
    add_edge_constraints(model, x, start_node, end_node)

    # Connectivity constraints
    for row in graph:
        for node in row:
            if node == start_node or node == end_node:
                continue
            model.add_constraint(
                model.sum(x[(node, neighbor)] for neighbor in node.neighbors.values() if (node, neighbor) in x) ==
                model.sum(x[(neighbor, node)] for neighbor in node.neighbors.values() if (neighbor, node) in x)
            )

    # No sub-tours
    for row in graph:
        for node in row:
            if node.is_obstacle or node == start_node or node == end_node:
                continue
            for neighbor in node.neighbors.values():
                if (node, neighbor) in x:
                    model.add_constraint(x[(node, neighbor)] + x[(neighbor, node)] <= 1)

    solution = model.solve()
    if solution:
        path = []
        # print("\nPath found with total cost:", model.objective_value)
        for e in x:
            if x[e].solution_value > 0.5:
                # print(f"{e[0].position} -> {e[1].position}")
                path.append(e[0])
        path.append(graph.objective)
        return path
    else:
        # print("No solution found")
        return None


def add_edge_constraints(model, x, start_node, end_node):
    # Helper function to add a single constraint
    def add_single_constraint(node, is_start, is_outgoing):
        neighbor_list = node.neighbors.values()
        if is_outgoing:
            edge_expr = model.sum(x[(node, neighbor)] for neighbor in neighbor_list if (node, neighbor) in x)
            value = 1 if is_start else 0
        else:
            edge_expr = model.sum(x[(neighbor, node)] for neighbor in neighbor_list if (neighbor, node) in x)
            value = 0 if is_start else 1
        
        return edge_expr == value
    
    # Constraints for start_node
    model.add_constraint(add_single_constraint(start_node, is_start=True, is_outgoing=True), "outgoing_edge_from_start")
    model.add_constraint(add_single_constraint(start_node, is_start=True, is_outgoing=False), "incoming_edge_to_start")

    # Constraints for end_node
    model.add_constraint(add_single_constraint(end_node, is_start=False, is_outgoing=True), "outgoing_edge_from_end")
    model.add_constraint(add_single_constraint(end_node, is_start=False, is_outgoing=False), "incoming_edge_to_end")
