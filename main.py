from Graph import Graph
from Node import Node


def main():
    """Create a graph from the given file and display it."""
    file_path = "examples/exo1.txt"
    graph = Graph(file_path)
    # graph.display()    # TODO: Implement this method : use display_nodes from utils/display.py
    graph.set_start(Node((0, 0)))
    graph.set_objective(Node((3, 8)))
    # graph.solve()     # TODO: Implement this method : use A* algorithm to find the shortest path


if __name__ == '__main__':
    main()
