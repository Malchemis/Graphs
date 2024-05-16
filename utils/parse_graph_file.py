import numpy as np
from pathlib import Path

from utils.convert import convert_to_node_graph


def parse_graph(path: Path):
    with open(path) as f:
        lines = f.readlines()
        n, m = map(int, lines[0].split())
        graph = np.zeros((n, m))

        index = 0
        for line in lines[1:]:
            parsed_line = line.split(" ")
            if parsed_line[-1] == "\n":
                parsed_line.remove("\n")
            graph[index] = np.array(list(map(int, parsed_line)))
            index += 1
    return convert_to_node_graph(graph)
