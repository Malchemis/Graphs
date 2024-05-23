import math
import warnings

from typing import Optional, Tuple, List, Dict
from pathlib import Path

from Node import Node
from utils.parse_graph_file import parse_graph, parse_tsp
from utils.constants import Directions as Dirs, Algorithms, Problems
from utils.distances import euclidean
from shortest_path_search import a_star, shortest_path_cplex_solver
from tsp_search import brute_force, tsp_cplex_solver


class Graph:
    """
    A graph class with basic non-oriented graph operations.
    X-axis is the vertical axis, Y-axis is the horizontal axis, with (0,0) in the top left corner.
    """

    def __init__(self, file_path: Optional[str] = None, problem: Optional[str] = None):
        self.start = None
        self.objective = None
        self.graph = []  # List[List[Node]] (mostly for shortest path)
        self.node_list = []  # List[int] (mostly for TSP)
        self.cost = {}   # Dict[Tuple[int, int], int] (for CPLEX)
        self.shape = (0, 0)
        self.file_path = file_path
        self.problem = problem
        if file_path and problem:
            if problem == Problems.TSP:
                self.node_list, self.cost = parse_tsp(Path(file_path))
            elif problem == Problems.SHORTEST_PATH:
                self.start, self.objective, self.graph = parse_graph(Path(file_path))
                self.shape = self.get_shape()
                self.cost = {edge: euclidean(edge[0], edge[1]) for edge in self.get_edges()}

    def set_start(self, node: Node):
        self.start = node

    def set_objective(self, node: Node):
        self.objective = node

    # Graph operations and methods
    def add_node(self, position: Tuple[int, int] | int):
        x, y = position
        if self.problem == Problems.SHORTEST_PATH:
            # The Graph needs to be extended if the position is out of bounds
            if x < 0 or x >= self.shape[0] or y < 0 or y >= self.shape[1]:
                self.extend_graph(position)
                self.shape = self.get_shape()   # Update the shape
            # The Node can be added if the position is empty
            if self.graph[x][y].is_obstacle:
                for neighbor in self.get_neighbors(position).values():
                    neighbor.add_neighbor(self.graph[x][y])
                self.graph[x][y].is_obstacle = False
                self.shape = self.get_shape()   # Update the shape
        elif self.problem == Problems.TSP:
            if position not in self.node_list:
                self.node_list.append(position)

    def add_edge(self, node1: Node | int, node2: Node | int, cost: int):
        if self.problem == Problems.TSP:
            if node1 not in self.graph:
                self.add_node(node1.position)
            if node2 not in self.graph:
                self.add_node(node2.position)
            if abs(node1.position[0] - node2.position[0]) + abs(node1.position[1] - node2.position[1]) <= math.sqrt(2):
                node1.add_neighbor(node1)
                node2.add_neighbor(node2)
                self.cost[(node1, node2)] = cost
            else:
                warnings.warn(f"Nodes {node1.position} and {node2.position} are not neighbors.")
        elif self.problem == Problems.SHORTEST_PATH:
            if node1 not in self.node_list:
                self.add_node(node1)
            if node2 not in self.node_list:
                self.add_node(node2)
            self.cost[(node1, node2)] = cost

    def remove_node(self, position: Tuple[int, int] | int):
        if self.problem == Problems.SHORTEST_PATH:
            self.graph[position[0]][position[1]].is_obstacle = True
            # disconnect the node from its neighbors
            for neighbor in self.get_neighbors(position).values():
                neighbor.neighbors.pop((neighbor.position[0] - position[0], neighbor.position[1] - position[1]))
            self.clean()
        elif self.problem == Problems.TSP:
            if position in self.node_list:
                self.node_list.remove(position)

    def solve(self, algo: Algorithms = Algorithms.A_STAR) -> Optional[List[Node]]:
        if self.problem == Problems.SHORTEST_PATH:
            if self.start is None or self.objective is None:
                warnings.warn("Start or objective node not set.")
                return
            if algo == Algorithms.A_STAR:
                path = a_star(self.start, self.objective)
                # Reset the nodes
                for row in self.graph:
                    for node in row:
                        node.g = float("inf")
                        node.h = 0
                        node.f = 0
                        node.parent = None
            elif algo == Algorithms.CPLEX:
                path = shortest_path_cplex_solver(self, self.start, self.objective)
            else:
                warnings.warn(f"Algorithm {algo} not implemented.")
                return
            return path
        elif self.problem == Problems.TSP:
            if algo == Algorithms.BRUTE_FORCE:
                path = brute_force(self)
            elif algo == Algorithms.CPLEX:
                path = tsp_cplex_solver(self)
            else:
                warnings.warn(f"Algorithm {algo} not implemented.")
                return
            return path

    def solve_and_save(self, strategy: Algorithms = Algorithms.A_STAR):
        if self.file_path is None:
            raise ValueError("Couldn't save the solution: File path not set.")
        path = self.solve(strategy)

        file_name = self.file_path.split("/")[-1].split(".")[0]
        text_to_save = ""

        if path is None:
            print("No path found.")
            return

        for node in path:
            if self.problem == Problems.SHORTEST_PATH:
                text_to_save += f"{node.position[0]} {node.position[1]}\n"
            else:
                text_to_save += f"{node}\n"

        # Create folder if it doesn't exist
        Path("solutions").mkdir(parents=True, exist_ok=True)

        with open(f"solutions/sol_{file_name}.txt", "w") as f:
            f.write(text_to_save)
            print(f"Solution saved in solutions/sol_{file_name}.txt")

    # -- Utils --
    def extend_graph(self, position: Tuple[int, int]):
        """Shift or add columns and lines to the graph to include the position."""
        if position[0] < 0:  # shift down
            self.graph.insert(0, [Node((0, j)) for j in range(len(self.graph[0]))])
        if position[1] < 0:  # shift right
            for row in self.graph:
                row.insert(0, Node((row[0].position[0], 0)))
        if position[0] >= len(self.graph):  # add row
            self.graph.append([Node((position[0], j)) for j in range(len(self.graph[0]))])
        if position[1] >= len(self.graph[0]):  # add column
            for row in self.graph:
                row.append(Node((row[0].position[0], position[1])))

    def clean(self):
        """Remove empty rows and columns from the graph."""
        self.graph = [row for row in self.graph if any(not node.is_obstacle for node in row)]
        self.shape = self.get_shape()

    def get_neighbors(self, position: Tuple[int, int]) -> Dict[Tuple[int, int], Node]:
        """Return a list of neighbors for a given node."""
        x, y = position
        neighbors = {}
        maze_height, maze_width = self.shape

        for dx, dy in [Dirs.N, Dirs.S, Dirs.E, Dirs.W, Dirs.NE, Dirs.SE, Dirs.SW, Dirs.NW]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < maze_height and 0 <= new_y < maze_width and not self.graph[new_x][new_y].is_obstacle:
                neighbors[(dx, dy)] = self.graph[new_x][new_y]
        return neighbors

    def get_edges(self):
        edges = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.graph[i][j].is_obstacle:
                    continue
                neighbors = self.get_neighbors((i, j))
                for neighbor in neighbors:
                    if neighbors[neighbor].is_obstacle:
                        continue
                    edges.append((self.graph[i][j], neighbors[neighbor]))
        return edges

    def get_shape(self):
        if self.problem == Problems.TSP:
            return len(self.node_list), None
        elif self.problem == Problems.SHORTEST_PATH:
            return len(self.graph), len(self.graph[0])
        raise ValueError("Problem not set.")

    def __getitem__(self, item):
        if self.problem == Problems.TSP:
            return self.node_list[item]
        return self.graph[item]

    def __setitem__(self, key, value):
        if self.problem == Problems.TSP:
            self.node_list[key] = value
        self.graph[key] = value

    def __len__(self):
        if self.problem == Problems.TSP:
            return len(self.node_list)
        return len(self.graph)

    def __iter__(self):
        if self.problem == Problems.TSP:
            return iter(self.node_list)
        return iter(self.graph)

    def __str__(self):
        if self.problem == Problems.TSP:
            return str(self.node_list)
        elif self.problem == Problems.SHORTEST_PATH:
            string_graph = "Start: " + str(self.start) + "\nObjective: " + str(self.objective) + "\n"
            for i in range(len(self.graph)):
                for j in range(len(self.graph[0])):
                    string_graph += f"{self.graph[i][j]} \n"
            return string_graph
        return "Empty graph."
