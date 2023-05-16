if __name__ == '__main__':
    edges = [
        ((0, 4), 6),
        ((0, 20), 2),
        ((0, 21), 4),
        ((1, 16), 10),
        ((1, 19), 1),
        ((1, 23), 2),
        ((2, 9), 3),
        ((2, 16), 6),
        ((2, 18), 1),
        ((3, 5), 10),
        ((3, 7), 8),
        ((3, 23), 7),
        ((4, 20), 3),
        ((4, 22), 5),
        ((5, 13), 5),
        ((5, 14), 9),
        ((6, 9), 6),
        ((6, 15), 10),
        ((6, 19), 8),
        ((7, 12), 3),
        ((7, 13), 3),
        ((8, 11), 3),
        ((8, 13), 9),
        ((8, 21), 8),
        ((9, 10), 9),
        ((10, 12), 6),
        ((10, 16), 8),
        ((11, 14), 6),
        ((11, 23), 4),
        ((12, 20), 2),
        ((14, 17), 5),
        ((15, 17), 5),
        ((15, 21), 7),
        ((17, 22), 4),
        ((18, 19), 3),
        ((18, 22), 1),
    ]

    num_nodes = 24
    state = [0] * num_nodes

    totSum = sum(v for _, v in edges)
    maxSum = 0
    minSum = totSum

    for i in range(2 << num_nodes):
        curr = 0
        for (e1, e2), val in edges:
            if state[e1] != state[e2]:
                curr += val
        maxSum = max(maxSum, curr)
        minSum = min(minSum, curr)

        ind = num_nodes - 1
        state[ind] += 1
        while ind >= 0 and state[ind] > 1:
            state[ind] = 0
            ind -= 1
            if ind >= 0:
                state[ind] += 1

    print(f"bounds: {totSum + 1 - maxSum}, {totSum + 1 - minSum}")
