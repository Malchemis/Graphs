from Graph import Graph
from utils.constants import Strategies
from tqdm import tqdm
import time


def main(n_iter=100, algo=Strategies.A_STAR, file_path="examples/reseau_50_50_1.txt", display=False, verbose=False,
         save=False):
    """Create a graph from the given file and display it."""
    graph = Graph(file_path)
    times = []

    for _ in tqdm(range(n_iter), desc=f"Running {algo} algorithm"):
        onset = time.perf_counter()
        path = graph.solve(algo)
        times.append(time.perf_counter() - onset)

        if display:
            graph.display(path)
            graph.display_network(path)
        if save:
            graph.solve_and_save(algo)

    print(f"Average time: {sum(times) / n_iter:.6f} seconds") if verbose else None


if __name__ == '__main__':
    main(verbose=True, algo=Strategies.CPLEX, n_iter=1, display=True, save=False, file_path="examples/reseau_50_50_1.txt")
