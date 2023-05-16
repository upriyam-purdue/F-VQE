from random import randint
from typing import Final, List, Tuple, Set


def _main():
    NUM_NODES: Final[int] = 10
    NUM_EDGES_PER_NODE: Final[int] = 3

    assert (NUM_EDGES_PER_NODE * NUM_NODES) % 2 == 0
    NUM_EDGES: Final[int] = NUM_EDGES_PER_NODE * NUM_NODES // 2

    node_set: Final[List[int]] = [NUM_EDGES_PER_NODE for _ in range(NUM_NODES)]
    edges: Final[List[Tuple[Tuple[int, int], float]]] = []
    for i in range(NUM_NODES):
        if node_set[i] > 0:
            num_edges_to_add: int = node_set[i]
            node_set[i] = 0

            nodes_to_add: List[int] = [i for i in range(NUM_NODES) if node_set[i] > 0]
            while len(nodes_to_add) > num_edges_to_add:
                nodes_to_add.pop(randint(0, len(nodes_to_add) - 1))

            for n in nodes_to_add:
                edges.append(((i, n), randint(1, 10)))
                print(f"{edges[-1]},")
                node_set[n] -= 1

    assert len(edges) == NUM_EDGES


if __name__ == '__main__':
    _main()
