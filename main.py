from Graph import Graph
from utils.constants import Strategies, Problems
from utils.display import display_cv2, display_network
from tqdm import tqdm
import time


def main(n_iter=100, problem=Problems.SHORTEST_PATH, algo=Strategies.A_STAR, file_path="examples/reseau_50_50_1.txt", display=False, verbose=False,
         save=False):
    """Create a graph from the given file and display it."""
    graph = Graph(file_path, problem)
    times = []

    for _ in tqdm(range(n_iter), desc=f"Running {algo} algorithm"):
        onset = time.perf_counter()
        path = graph.solve(algo)
        times.append(time.perf_counter() - onset)

        if verbose:
            print(f"Path: {path}")
        if display:
            display_cv2(graph, path)
            display_network(graph, path)
        if save:
            graph.solve_and_save(algo)

    print(f"Average time: {sum(times) / n_iter:.6f} seconds") if verbose else None
    return graph


def compare_shortest_path_algo(n_iter=100, file_path="examples/reseau_50_50_1.txt"):
    """Compare the execution time of the A* and CPLEX algorithms."""
    main(n_iter=n_iter, algo=Strategies.A_STAR, file_path=file_path, verbose=True)
    main(n_iter=n_iter, algo=Strategies.CPLEX, file_path=file_path, verbose=True)


if __name__ == '__main__':
    main(verbose=True, problem=Problems.TSP, algo=Strategies.CPLEX, n_iter=1, display=True, save=True,
         file_path="examples/tsp1.txt")
    # compare_shortest_path_algo(n_iter=10, file_path="examples/reseau_50_50_1.txt")
