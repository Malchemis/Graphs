"""
Traveling Salesman Problem (TSP) search algorithms
- Using Brute force
- Using CPLEX solver
"""

from typing import List, Optional

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

    # Generate all possible routes that start and end from any node
    for starting_ending_point in nodes:
        # get all permutations of the nodes but the starting and ending point
        for perm in permutations([node for node in nodes if node != starting_ending_point]):
            route = (starting_ending_point,) + perm + (starting_ending_point,)

            # Calculate the total cost of this route
            current_cost = 0
            valid_route = True
            for i in range(len(route) - 1):
                if (route[i], route[i + 1]) in costs:
                    current_cost += costs[(route[i], route[i + 1])]
                else:
                    valid_route = False
                    break

            # Check if this route is cheaper and valid
            if valid_route and current_cost < min_cost:
                min_cost = current_cost
                best_route = route

    return best_route


def tsp_cplex_solver(graph) -> Optional[List]:
    pass
