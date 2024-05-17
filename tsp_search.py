"""
Traveling Salesman Problem (TSP) search algorithms
- Using Brute force
- Using CPLEX solver
"""

from typing import List, Optional
import time
from docplex.mp.model import Model

from itertools import permutations


def brute_force(graph) -> tuple:
    """
    Brute force algorithm to solve the Traveling Salesman Problem (TSP)
    :param graph: the graph to explore
    :return: the shortest path found
    """
    nodes = graph.node_list
    costs = graph.cost

    min_cost = float('inf')
    best_route = []

    # Generate all possible routes that start and end at the first node in the list
    for perm in permutations(nodes[1:]):
        # Start at the first node and add it to the end to complete the circuit
        route = (nodes[0],) + perm + (nodes[0],)

        # Calculate the total cost of this route
        current_cost = 0
        valid_route = True
        for i in range(len(route) - 1):
            if (route[i], route[i + 1]) in costs:
                current_cost += costs[(route[i], route[i + 1])]
            else:
                valid_route = False
                break

        print(f"Route: {route} - Cost: {current_cost}")

        # Check if this route is cheaper and valid
        if valid_route and current_cost < min_cost:
            print(f"New best route: {route} - Cost: {current_cost}")
            min_cost = current_cost
            best_route = route

    return best_route, min_cost


def tsp_cplex_solver(graph) -> Optional[List]:
    pass
