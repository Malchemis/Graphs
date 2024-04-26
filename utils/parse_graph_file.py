import numpy as np
from pathlib import Path

from Node import Node
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


def save_graph(graph: list[list[Node | None]], file_path: Path) -> None:
    with open(file_path, 'w') as f:
        f.write(f'{len(graph)} {len(graph[0])}\n')
        for row in graph:
            f.write(' '.join(map(str, row)) + '\n')
