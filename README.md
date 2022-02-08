# F-VQEs
See [https://arxiv.org/pdf/2106.10055.pdf](https://arxiv.org/pdf/2106.10055.pdf).

## How to Use
Just run the main code (calls `_test_circuit_training`).
This will train the F-VQE on the simple MaxCut problem provided. 

The solution to the default problem (defined in `_APPLY_PROBLEM_COST_FUNCTION`) is simply `(0,4),(1,2,3)`, with a total score of `20`.
This corresponds to a target score of `1.5` once the MaxCut score is translated into a minimization problem.
Therefore, you should see the F-VQE's iteration score (printed as the circuit trains) decrease from its initial value (around 11 or 12, typically) to a final result near 1.5.

## How to Use 2
To define other problems, simply convert the problem into a minimization problem, where the cost/success score is always positive.
Simply code the cost function into `APPLY_PROBLEM_COST_FUNCTION` and run the training.
