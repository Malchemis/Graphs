import cv2
import numpy as np

from Graph import Graph

WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
START_COLOR = (2, 159, 0)
END_COLOR = (74, 120, 255)


def display_nodes(graph: Graph):
    img = np.zeros((len(graph), len(graph[0]), 3))

    for i in range(len(img)):
        for j in range(len(img[0])):
            if graph[i][j] is None:
                img[i][j] = BLACK_COLOR
            elif graph[i][j] == graph.start:
                img[i][j] = START_COLOR
            elif graph[i][j] == graph.objective:
                img[i][j] = END_COLOR
            else:
                img[i][j] = WHITE_COLOR

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (len(img) * 50, len(img[0]) * 50), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Graph", img)
    key = cv2.waitKey(10)
    if key == ord("q"):
        cv2.destroyAllWindows()
        return


def display_network(graph : Graph):
    pass