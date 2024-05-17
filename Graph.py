import math
import warnings
import networkx as nx
import numpy as np
import cv2
import matplotlib.pyplot as plt

from typing import Optional, Tuple, List, Dict
from pathlib import Path

from Node import Node
from utils.parse_graph_file import parse_graph, parse_tsp
from utils.constants import Directions as Dirs, Colors, Strategies, Problems
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
        self.graph = []  # List[List[Node]] if problem is shortest path
        self.node_list = []  # List[Node] if problem is TSP
        self.cost = {}   # Dict[Tuple[int, int], int] if problem is TSP
        self.shape = (0, 0)
        self.file_path = file_path
        self.problem = problem
        if file_path and problem:
            if problem == Problems.TSP:
                self.node_list, self.cost = parse_tsp(Path(file_path))
            elif problem == Problems.SHORTEST_PATH:
                self.start, self.objective, self.graph = parse_graph(Path(file_path))
                self.shape = self.get_shape()

    def set_start(self, node: Node):
        self.start = node

    def set_objective(self, node: Node):
        self.objective = node

    # Graph operations and methods
    def add_node(self, position: Tuple[int, int] | int):
        if self.problem == Problems.SHORTEST_PATH:
            # The Graph needs to be extended if the position is out of bounds
            if position[0] < 0 or position[0] >= len(self.graph) or position[1] < 0 or position[1] >= len(self.graph[0]):
                self.graph = self.extend_graph(position)
                self.shape = self.get_shape()
            # The Node can be added if the position is empty
            if self.graph[position[0]][position[1]].is_obstacle:
                self.graph[position[0]][position[1]].neighbors = self.get_neighbors(position)
                self.graph[position[0]][position[1]].is_obstacle = False
                self.shape = self.get_shape()
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
                node1.add_neighbor(node2, cost)
                node2.add_neighbor(node1, cost)
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
            self.clean()
        elif self.problem == Problems.TSP:
            if position in self.node_list:
                self.node_list.remove(position)

    def solve(self, strategy: Strategies = Strategies.A_STAR) -> Optional[List[Node]]:
        if self.problem == Problems.SHORTEST_PATH:
            if self.start is None or self.objective is None:
                warnings.warn("Start or objective node not set.")
                return
            if strategy == Strategies.A_STAR:
                path = a_star(self.start, self.objective)
                # Reset the nodes
                for row in self.graph:
                    for node in row:
                        node.g = float("inf")
                        node.h = 0
                        node.f = 0
                        node.parent = None
            elif strategy == Strategies.CPLEX:
                path = shortest_path_cplex_solver(self, self.start, self.objective)
            else:
                warnings.warn(f"Strategy {strategy} not implemented.")
                return
            return path
        elif self.problem == Problems.TSP:
            if strategy == Strategies.BRUTE_FORCE:
                path = brute_force(self)
            elif strategy == Strategies.CPLEX:
                path = tsp_cplex_solver(self)
            else:
                warnings.warn(f"Strategy {strategy} not implemented.")
                return
            return path

    def solve_and_save(self, strategy: Strategies = Strategies.A_STAR):
        if self.file_path is None:
            raise ValueError("Couldn't save the solution: File path not set.")
        path = self.solve(strategy)

        if path is None:
            print("Couldn't save the solution: No path found.")
            return

        file_name = self.file_path.split("/")[-1].split(".")[0]
        text_to_save = ""

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
    def extend_graph(self, position: Tuple[int, int]) -> List[List[Node]]:
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
        return self.graph

    def clean(self):
        """Remove empty rows and columns from the graph."""
        self.graph = [row for row in self.graph if any(not node.is_obstacle for node in row)]
        self.shape = self.get_shape()

    def get_neighbors(self, position: Tuple[int, int]) -> Dict[Tuple[int, int], Tuple[int, Node]]:
        """Return a neighbors for a given node."""
        x, y = position
        neighbors = {}
        maze_height, maze_width = self.shape
        # North, South, East, West
        for dx, dy in [Dirs.N, Dirs.S, Dirs.E, Dirs.W]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < maze_height and 0 <= new_y < maze_width and not self.graph[new_x][new_y].is_obstacle:
                neighbors[(dx, dy)] = (1, self.graph[new_x][new_y])
        # North - East, South - East, South - West, North - West
        for dx, dy in [Dirs.NE, Dirs.SE, Dirs.SW, Dirs.NW]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < maze_height and 0 <= new_y < maze_width and not self.graph[new_x][new_y].is_obstacle:
                neighbors[(dx, dy)] = (math.sqrt(2), self.graph[new_x][new_y])
        return neighbors

    def get_edges(self):
        edges = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.graph[i][j].is_obstacle:
                    continue
                neighbors = self.get_neighbors((i, j))
                for neighbor in neighbors:
                    if neighbors[neighbor][1].is_obstacle:
                        continue
                    edges.append((self.graph[i][j], neighbors[neighbor][1]))
        return edges

    def display_network(self, path: Optional[List[Node]] = None):
        """
        Display the graph as a networkx graph.
        :return: weighted graph (nx graph)
        """
        nx_G = None
        if self.problem == Problems.SHORTEST_PATH:
            nx_G = self.display_network_shortest_path(path)
        elif self.problem == Problems.TSP:
            nx_G = self.display_network_tsp(path)
        if nx_G:
            node_color_map = nx.get_node_attributes(nx_G, 'color').values()
            edge_color_map = nx.get_edge_attributes(nx_G, 'color').values()
            pos = nx.get_node_attributes(nx_G, 'pos')
            nx.draw_networkx(nx_G, with_labels=False, pos=pos, node_color=node_color_map, edge_color=edge_color_map)
            nx.draw_networkx_labels(nx_G, pos, font_size=10, font_color='white')
            plt.show()
        else:
            raise ValueError("Problem not set.")

    def display_network_tsp(self, path: Optional[List[Node]] = None):
        # Convert the graph to a networkx graph
        nx_G = nx.Graph()
        for i in range(len(self.node_list)):
            node_id = i
            # Set their position in a circle
            pos_i = (math.cos(2 * math.pi * i / len(self.node_list)), math.sin(2 * math.pi * i / len(self.node_list)))
            nx_G.add_node(node_id, pos=pos_i, color="#1f78b4")

            for j in range(i + 1, len(self.node_list)):
                neighbor_id = j
                if nx_G.has_edge(neighbor_id, node_id):
                    continue

                pos_neighbor = (math.cos(2 * math.pi * j / len(self.node_list)), math.sin(2 * math.pi * j / len(self.node_list)))
                nx_G.add_node(neighbor_id, pos=pos_neighbor, color="#1f78b4")

                if (node_id, neighbor_id) in self.cost:
                    nx_G.add_edge(node_id, neighbor_id, weight=self.cost[(self.node_list[i], self.node_list[j])], color="black")

        # Display the graph
        plt.figure(figsize=(len(self.node_list)/2, len(self.node_list)/2))
        return nx_G

    def display_network_shortest_path(self, path: Optional[List[Node]] = None):
        # Convert the graph to a networkx graph
        nx_G = nx.Graph()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.graph[i][j].is_obstacle:
                    continue
                neighbors = self.get_neighbors((i, j))
                node_id = i * self.shape[1] + j
                for neighbor in neighbors:
                    neighbor_id = (i + neighbor[0]) * self.shape[1] + j + neighbor[1]
                    # Add nodes
                    if nx_G.has_edge(neighbor_id, node_id):
                        continue
                    # Find color
                    color = "#1f78b4"
                    color_neighbor = "#1f78b4"
                    edge_color = "black"
                    if path is not None:
                        # Path color
                        if self.graph[i][j] in path:
                            if self.graph[i][j] == self.start:
                                color = "#029f00"
                            elif self.graph[i][j] == self.objective:
                                color = "#4a78ff"
                            else:
                                color = "red"
                        if self.graph[i + neighbor[0]][j + neighbor[1]] in path:
                            if self.graph[i + neighbor[0]][j + neighbor[1]] == self.start:
                                color_neighbor = "#029f00"
                            elif self.graph[i + neighbor[0]][j + neighbor[1]] == self.objective:
                                color_neighbor = "#4a78ff"
                            else:
                                color_neighbor = "red"
                        if color != "#1f78b4" and color_neighbor != "#1f78b4":
                            edge_color = "red"

                    nx_G.add_node(node_id, pos=(j, len(self.graph)-i), color=color)
                    nx_G.add_node(neighbor_id, pos=(j + neighbor[1], len(self.graph) - (i + neighbor[0])), color=color_neighbor)
                    nx_G.add_edge(node_id, neighbor_id, weight=neighbors[neighbor][0], color=edge_color)

        # Display the graph
        plt.figure(figsize=(len(self.graph[0])/2, len(self.graph)/2))
        return nx_G

    def display(self, path: Optional[List[Node]] = None):
        """Display the graph in a window. Press 'q' to close."""
        img = None
        if self.problem == Problems.TSP:
            img = self.display_cv2_tsp(path)
        elif self.problem == Problems.SHORTEST_PATH:
            img = self.display_cv2_shortest_path(path)
        if img is not None:
            while len(img) > 1000 or len(img[0]) > 1900:
                img = cv2.resize(img, (len(img[0]) // 2, len(img) // 2), interpolation=cv2.INTER_NEAREST)

            cv2.imshow("Graph", img)
            key = cv2.waitKey(10)

            # Wait for 'q' key to close
            while key != ord("q"):
                key = cv2.waitKey(10)
            cv2.destroyAllWindows()
        else:
            raise ValueError("Problem not set.")

    def display_cv2_tsp(self, path: Optional[List[Node]] = None):
        # Setup
        img = np.zeros((len(self.node_list), len(self.node_list), 3), dtype=np.uint8)

        # Display
        for i in range(len(img)):
            for j in range(len(img[0])):
                if self.node_list[i] in path:
                    img[i][j] = Colors.PATH_COLOR
                else:
                    img[i][j] = Colors.WHITE_COLOR

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (len(self.node_list) * 50, len(self.node_list) * 50), interpolation=cv2.INTER_NEAREST)
        return img

    def display_cv2_shortest_path(self, path: Optional[List[Node]] = None):
        # Setup
        img = np.zeros((len(self.graph), len(self.graph[0]), 3), dtype=np.uint8)

        # Display
        for i in range(len(img)):
            for j in range(len(img[0])):
                if self.graph[i][j].is_obstacle:
                    img[i][j] = Colors.BLACK_COLOR
                elif self.graph[i][j] == self.start:
                    img[i][j] = Colors.START_COLOR
                elif self.graph[i][j] == self.objective:
                    img[i][j] = Colors.END_COLOR
                else:
                    if path is not None:
                        if self.graph[i][j] in path:
                            img[i][j] = Colors.PATH_COLOR
                        else:
                            img[i][j] = Colors.WHITE_COLOR
                    else:
                        img[i][j] = Colors.WHITE_COLOR

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (len(self.graph[0]) * 50, len(self.graph) * 50), interpolation=cv2.INTER_NEAREST)
        return img

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
