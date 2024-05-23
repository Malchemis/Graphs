from Graph import Graph
from utils.constants import Strategies, Problems
from utils.display import display_cv2, display_network
from utils.generation import gen_tsp
from tqdm import tqdm
import time


def main(n_iter=100, problem=Problems.SHORTEST_PATH, algo=Strategies.A_STAR, file_path="examples/reseau_50_50_1.txt", display=True, verbose=False,
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

    print(f"Average time: {sum(times) / n_iter:.6f} seconds")
    return graph


def compare_shortest_path_algo(n_iter=100, problem=Problems.SHORTEST_PATH, file_path="examples/reseau_50_50_1.txt"):
    """Compare the execution time of the A* and CPLEX algorithms."""
    if problem == Problems.TSP:
        main(n_iter=n_iter, algo=Strategies.BRUTE_FORCE, file_path=file_path, problem=Problems.TSP)
        main(n_iter=n_iter, algo=Strategies.CPLEX, file_path=file_path, problem=Problems.TSP)
    elif problem == Problems.SHORTEST_PATH:
        main(n_iter=n_iter, algo=Strategies.A_STAR, file_path=file_path)
        main(n_iter=n_iter, algo=Strategies.CPLEX, file_path=file_path)


if __name__ == '__main__':
    gen_tsp(30, .2)
    main(verbose=True, problem=Problems.TSP, algo=Strategies.BRUTE_FORCE, n_iter=1, display=True, save=True,
    file_path="examples/tsp.txt")
    # compare_shortest_path_algo(n_iter=100, file_path="examples/tsp1.txt", problem=Problems.TSP)
