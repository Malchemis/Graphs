import random
from pathlib import Path


def gen_tsp(n: int, p:float, file_name:str = "tsp.txt"):
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

    print(f"File saved in examples/{file_name}")