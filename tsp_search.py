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
    :return: the cheapest path found
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
    """
    Solve the Traveling Salesman Problem (TSP) using the CPLEX solver
    :param graph: the graph to explore
    :return: the cheapest path found
    """
    nodes = graph.node_list
    costs = graph.cost

    # Create a new model
    mdl = Model('TSP')

    # Create variables: x[i, j] is 1 if edge (i, j) is part of the solution
    x = mdl.binary_var_dict(costs.keys(), name='x')

    # Objective: Minimize the total cost of the tour
    mdl.minimize(mdl.sum(x[i, j] * costs[i, j] for (i, j) in costs))

    # Constraints: Each node must be entered and left exactly once
    # Unique entering and leaving edges for each node
    for k in nodes:
        mdl.add_constraint(mdl.sum(x[i, k] for i in nodes if (i, k) in x) == 1, f'enter_{k}')
        mdl.add_constraint(mdl.sum(x[k, j] for j in nodes if (k, j) in x) == 1, f'leave_{k}')

    # No Sub tours
    for i in nodes:
        for j in nodes:
            if i != j and (i, j) in x:
                mdl.add_constraint(x[i, j] + x[j, i] <= 1, f'sub_tour_{i}_{j}')

    # Solve the model
    solution = mdl.solve()

    # Check if a solution exists
    if solution:
        print("Solution found:")
        edges = [(i, j) for i, j in x if x[i, j].solution_value > 0.5]
        print("Edges in the tour:", edges)
        print("Path", path := [edge[0] for edge in edges])
        return path + [edges[-1][1]]
    else:
        print("No solution exists.")
        return None
