from Graph import Graph
from utils.constants import Strategies


def main():
    """Create a graph from the given file and display it."""
    file_path = "examples/exo1.txt"
    graph = Graph(file_path)
    path = graph.solve(Strategies.A_STAR)
    #graph.display(path)
    graph.display_network(path)

if __name__ == '__main__':
    main()
