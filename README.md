# F-VQEs
A basic implementation of F-VQEs for application with black-box cost functions.

## Table of Contents
* [Background Reading](#background-reading)
* [How to Use](#how-to-use)
* [How to Use 2](#how-to-use-2)
* [Libraries Used](#libraries-used)

## Background Reading
See [https://arxiv.org/pdf/2106.10055.pdf](https://arxiv.org/pdf/2106.10055.pdf).

## How to Use
Just run the main code (calls `_test_circuit_training`).
This will train the F-VQE on the simple MaxCut problem provided. 

The solution to the default problem (defined in `_APPLY_PROBLEM_COST_FUNCTION`) is simply `(0,4),(1,2,3)`, with a total score of `20`.
This corresponds to a target score of `1.5` once the MaxCut score is translated into a minimization problem.
Therefore, you should see the F-VQE's iteration score (printed as the circuit trains) decrease from its initial value (around 11 or 12, typically) to a final result near 1.5.

The default provided problem's cost function code:
```python
def _APPLY_PROBLEM_COST_FUNCTION(input_binary_values: Tuple[int, ...]) -> float:
    # sample maxcut problem for 5 nodes
    edges = [
        ((0, 1), 5),
        ((0, 2), 3),
        ((1, 3), 1),
        ((2, 4), 5),
        ((3, 4), 7)
    ]
    total_edge_cost = sum(cost for _, cost in edges)
    groups = [2 * binary_var - 1 for binary_var in input_binary_values]
    return total_edge_cost + 0.5 - sum(cost if groups[edge[0]] != groups[edge[1]] else 0 for edge, cost in edges)
```

## How to Use 2
To define other problems, simply convert the problem into a minimization problem, where the cost/success score is always positive.
Simply code the cost function into `APPLY_PROBLEM_COST_FUNCTION` and run the training.

## Libraries Used
* [tket]()
  * [numpy]()
  * [sympy]()
