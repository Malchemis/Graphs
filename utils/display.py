from typing import Optional, List

from Node import Node
from utils.constants import Colors, Problems

import networkx as nx
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


def display_cv2(graph, path: Optional[List[Node]] = None):
    """Display the graph in a window. Press 'q' to close."""
    img = None
    if graph.problem == Problems.TSP:
        # img = display_cv2_tsp(graph, path)
        pass
    elif graph.problem == Problems.SHORTEST_PATH:
        img = display_cv2_shortest_path(graph, path)
    if img is not None:
        while len(img) > 1000 or len(img[0]) > 1900:
            img = cv2.resize(img, (len(img[0]) // 2, len(img) // 2), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Graph", img)
        key = cv2.waitKey(10)

        # Wait for 'q' key to close
        while key != ord("q"):
            key = cv2.waitKey(10)
        cv2.destroyAllWindows()
    if not graph.problem:
        raise ValueError("Problem not set.")


def display_network(graph, path: Optional[List[Node]] = None):
    """
    Display the graph as a networkx graph.
    :return: weighted graph (nx graph)
    """
    nx_graph = None
    if graph.problem == Problems.SHORTEST_PATH:
        nx_graph = display_network_shortest_path(graph, path)
    elif graph.problem == Problems.TSP:
        nx_graph = display_network_tsp(graph, path)
    if nx_graph:
        node_color_map = nx.get_node_attributes(nx_graph, 'color').values()
        edge_color_map = nx.get_edge_attributes(nx_graph, 'color').values()
        pos = nx.get_node_attributes(nx_graph, 'pos')
        nx.draw_networkx(nx_graph, with_labels=False, pos=pos, node_color=node_color_map, edge_color=edge_color_map)
        nx.draw_networkx_labels(nx_graph, pos, font_color="white")
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=nx.get_edge_attributes(nx_graph, 'weight'))
        plt.show()
    else:
        raise ValueError("Problem not set.")


def display_network_tsp(graph, path: Optional[List[Node]] = None):
    # Convert the graph to a networkx graph
    nx_graph = nx.Graph()
    for i in range(len(graph.node_list)):
        node_id = i
        # Set their position in a circle
        pos_i = (math.cos(2 * math.pi * i / len(graph.node_list)), math.sin(2 * math.pi * i / len(graph.node_list)))
        nx_graph.add_node(node_id, pos=pos_i, color="#1f78b4")

    # Add indexes in path order as edges
    if path is not None:
        for i in range(len(path) - 1):
            nx_graph.add_edge(graph.node_list.index(path[i]), graph.node_list.index(path[i + 1]),
                              weight=path.index(path[i]) + 1, color="red")

    # Display the graph
    plt.figure(figsize=(len(graph.node_list) / 2, len(graph.node_list) / 2))
    return nx_graph


def display_network_shortest_path(graph, path: Optional[List[Node]] = None):
    # Convert the graph to a networkx graph
    nx_graph = nx.Graph()
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if graph.graph[i][j].is_obstacle:
                continue
            neighbors = graph.get_neighbors((i, j))
            node_id = i * graph.shape[1] + j
            for neighbor in neighbors:
                neighbor_id = (i + neighbor[0]) * graph.shape[1] + j + neighbor[1]
                # Add nodes
                if nx_graph.has_edge(neighbor_id, node_id):
                    continue
                # Find color
                color = "#1f78b4"
                color_neighbor = "#1f78b4"
                edge_color = "black"
                if path is not None:
                    # Path color
                    if graph.graph[i][j] in path:
                        if graph.graph[i][j] == graph.start:
                            color = "#029f00"
                        elif graph.graph[i][j] == graph.objective:
                            color = "#4a78ff"
                        else:
                            color = "red"
                    if graph.graph[i + neighbor[0]][j + neighbor[1]] in path:
                        if graph.graph[i + neighbor[0]][j + neighbor[1]] == graph.start:
                            color_neighbor = "#029f00"
                        elif graph.graph[i + neighbor[0]][j + neighbor[1]] == graph.objective:
                            color_neighbor = "#4a78ff"
                        else:
                            color_neighbor = "red"
                    if color != "#1f78b4" and color_neighbor != "#1f78b4":
                        edge_color = "red"

                nx_graph.add_node(node_id, pos=(j, len(graph) - i), color=color)
                nx_graph.add_node(neighbor_id, pos=(j + neighbor[1], len(graph) - (i + neighbor[0])),
                                  color=color_neighbor)
                nx_graph.add_edge(node_id, neighbor_id, weight=neighbors[neighbor][0], color=edge_color)

    # Display the graph
    plt.figure(figsize=(len(graph[0]) / 2, len(graph) / 2))
    return nx_graph


def display_cv2_shortest_path(graph, path: Optional[List[Node]] = None):
    # Setup
    img = np.zeros((len(graph.graph), len(graph.graph[0]), 3), dtype=np.uint8)

    # Display
    for i in range(len(img)):
        for j in range(len(img[0])):
            if graph.graph[i][j].is_obstacle:
                img[i][j] = Colors.BLACK_COLOR
            elif graph.graph[i][j] == graph.start:
                img[i][j] = Colors.START_COLOR
            elif graph.graph[i][j] == graph.objective:
                img[i][j] = Colors.END_COLOR
            else:
                if path is not None:
                    if graph.graph[i][j] in path:
                        img[i][j] = Colors.PATH_COLOR
                    else:
                        img[i][j] = Colors.WHITE_COLOR
                else:
                    img[i][j] = Colors.WHITE_COLOR

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (len(graph[0]) * 50, len(graph) * 50), interpolation=cv2.INTER_NEAREST)
    return img
