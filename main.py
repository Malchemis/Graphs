from Graph import Graph
from utils.constants import Algorithms, Problems
from utils.display import display_cv2, display_network
from utils.generation import gen_tsp, gen_astar
from tqdm import tqdm
import time


def main(n_iter=100, problem=Problems.SHORTEST_PATH, algo=Algorithms.A_STAR, file_path="examples/reseau_50_50_1.txt",
         display=False, verbose=False, save=False, n=7, p=.3):
    """Create a graph from the given file and display it."""
    gen_tsp(n, p) if problem == Problems.TSP else gen_astar(n, p) if problem == Problems.SHORTEST_PATH else None
    graph = Graph(file_path, problem)
    times = []

    for _ in tqdm(range(n_iter), desc=f"Running {algo} algorithm"):
        onset = time.perf_counter()
        path = graph.solve(algo)
        times.append(time.perf_counter() - onset)

        if verbose:
            print(f"\nPath: {path}") if path else print("No path found")
        if display:
            display_cv2(graph, path)
            display_network(graph, path)
        if save:
            graph.solve_and_save(algo)

    print(f"\nAverage time for {n_iter} iterations with {algo}: {sum(times) / n_iter:.5f}s")
    return graph


def compare_algo(n_iter=100, problem=Problems.SHORTEST_PATH, file_path="examples/tsp.txt", n=7, p=.3):
    """Compare the execution time of the A* and CPLEX algorithms."""
    if problem == Problems.TSP:
        main(n_iter=n_iter, algo=Algorithms.CPLEX, file_path=file_path, problem=Problems.TSP, n=n, p=p)
        main(n_iter=n_iter, algo=Algorithms.BRUTE_FORCE, file_path=file_path, problem=Problems.TSP, n=n, p=p)
    elif problem == Problems.SHORTEST_PATH:
        main(n_iter=n_iter, algo=Algorithms.A_STAR, file_path=file_path, problem=Problems.SHORTEST_PATH, n=n, p=p)
        main(n_iter=n_iter, algo=Algorithms.CPLEX, file_path=file_path, problem=Problems.SHORTEST_PATH, n=n, p=p)


if __name__ == '__main__':
    compare_algo(n_iter=100, problem=Problems.TSP, file_path="examples/tsp.txt", n=7, p=.3)
    # compare_algo(n_iter=100, problem=Problems.SHORTEST_PATH, file_path="examples/astar.txt", n=50, p=.2)
