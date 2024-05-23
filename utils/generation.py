import random
from pathlib import Path

from utils.constants import Values


def gen_tsp(n: int, p: float, file_name: str = "tsp.txt", verbose: bool = False):
    """Generates a random TSP problem with n nodes and probability p of having an edge between nodes."""
    nodes = []
    cost = {}

    # Generate nodes
    for i in range(n):
        nodes.append(i)

    # Generate edges
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                cost[(i, j)] = random.randint(10, 50)
    # Write to file
    Path("examples").mkdir(parents=True, exist_ok=True)
    with open(f"examples/{file_name}", "w") as f:
        f.write(f"{n} {len(cost)}\n")
        for (i, j), c in cost.items():
            f.write(f"{i} {j} {c}\n")

    if verbose:
        print(f"File saved in examples/{file_name}")


def gen_astar(n: int, p: float, file_name: str = "astar.txt", verbose: bool = False):
    """Generates a random shortest path problem with n^2 nodes and probability p of having an edge between nodes."""
    graph = [[Values.WALL for _ in range(n)] for _ in range(n)]

    # Generate nodes
    for i in range(n):
        for j in range(n):
            if random.random() < p:
                graph[i][j] = Values.EMPTY

    # Start and objective. Select a random start and objective from the empty nodes
    empty_nodes = [(i, j) for i in range(n) for j in range(n) if graph[i][j] == Values.EMPTY]
    start, objective = random.sample(empty_nodes, 2)
    graph[start[0]][start[1]] = Values.START
    graph[objective[0]][objective[1]] = Values.OBJECTIVE

    # Write to file
    Path("examples").mkdir(parents=True, exist_ok=True)
    with open(f"examples/{file_name}", "w") as f:
        f.write(f"{n} {n}\n")
        for i in range(n):
            for j in range(n):
                f.write(f"{graph[i][j]} ")
            f.write("\n")

    if verbose:
        print(f"File saved in examples/{file_name}")
