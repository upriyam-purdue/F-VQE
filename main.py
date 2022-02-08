from typing import List, Dict, Tuple, Final, Callable

from pytket import Circuit
from pytket.circuit.display import render_circuit_as_html
from sympy import Symbol
from numpy import pi, sqrt
from pytket.extensions.qiskit import AerBackend
from pytket.passes import CommuteThroughMultis, RemoveRedundancies, SequencePass, RepeatWithMetricPass

# globals
_NUM_QUBITS: Final[int] = 5
_CIRCUIT_BLOCK_DEPTH: Final[int] = 1

_NUM_SHOTS_PER_CIRCUIT: Final[int] = 1000

_QUANTUM_BACKEND: Final[any] = AerBackend()


# noinspection PyPep8Naming
def _APPLY_ENERGY_FILTER(energy: float, tau: float) -> float:
    # inverse filter --> see paper (https://arxiv.org/pdf/2106.10055.pdf) Figure 1 (page 1), Figure 2 (page 2)
    return energy ** -tau


# noinspection PyPep8Naming
def _APPLY_PROBLEM_COST_FUNCTION(input_binary_values: Tuple[int, ...]) -> float:
    # TODO implement problem cost function -- should be a minimization problem
    # MUST RETURN POSITIVE VALUES ONLY
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


# helper functions
def _get_symbol_dict(symbol_defs: List[float]) -> Dict[str, float]:
    s_dict = {}

    # dump values from list to dictionary of values
    for ind, sdef in enumerate(symbol_defs):
        s_dict[Symbol(f"alpha{ind}")] = sdef

    # return dictionary
    return s_dict


# define helper functions
def _simplify_circuit_for_args(circuit: Circuit, param_values: List[float]) -> Circuit:
    # duplicate compiled circuit & replace variables
    circuit_simplified = circuit.copy()
    circuit_simplified.symbol_substitution(_get_symbol_dict(param_values))

    # simplify compiled/substituted circuit -- make sure selected simplifications are supported by _QUANTUM_BACKEND
    simplify_pass = SequencePass([RemoveRedundancies(), CommuteThroughMultis()])
    simplifier = SequencePass([simplify_pass, RepeatWithMetricPass(simplify_pass, lambda circ: circ.n_gates)])
    simplifier.apply(circuit_simplified)

    # return simplified circuit
    return circuit_simplified


# measure circuit with filter
def _measure_ansatz(compiled_circuit: Circuit, param_values: List[float],
                    square_filtered_output: bool = False, apply_filter: bool = True) -> float:
    # simplify circuit
    circuit = _simplify_circuit_for_args(compiled_circuit, param_values)

    # get results of circuit
    job_handle = _QUANTUM_BACKEND.process_circuit(circuit, _NUM_SHOTS_PER_CIRCUIT)  # submit the job to run the circuit
    counts = _QUANTUM_BACKEND.get_result(job_handle).get_counts()  # retrieve and summarise the results

    # compute value of tau --> see paper (https://arxiv.org/pdf/2106.10055.pdf) Section II.D (page 4)
    tau = 0.4  # TODO dynamically compute value of tau

    # process with Monte Carlo estimator --> see paper (https://arxiv.org/pdf/2106.10055.pdf) eq. 8 (page 4)
    def process_quantum_state(q_state: Tuple[int, ...]):
        energy = _APPLY_PROBLEM_COST_FUNCTION(q_state)
        if apply_filter:
            energy = _APPLY_ENERGY_FILTER(energy, tau)
            if square_filtered_output:
                energy = energy ** 2
        return energy

    return sum(count * process_quantum_state(q_state) for q_state, count in counts.items()) / _NUM_SHOTS_PER_CIRCUIT


# compute new params from old values
def _compute_new_params(circuit: Circuit, param_values: List[float]) -> List[float]:
    # memoize denominator value --> 4 * sqrt(f_squared)
    derivative_denominator = 4 * sqrt(_measure_ansatz(circuit, param_values, square_filtered_output=True))

    # compute numerator values
    derivative_numerators = []
    for ind in range(len(param_values)):
        param_values[ind] += pi / 2
        param_plus = _measure_ansatz(circuit, param_values)  # psi(j+)
        param_values[ind] -= pi
        param_minus = _measure_ansatz(circuit, param_values)  # psi(j-)
        param_values[ind] += pi / 2

        # psi(j+) - psi(j-) --> see paper (https://arxiv.org/pdf/2106.10055.pdf) eq. 6 (page 3)
        derivative_numerators.append(param_plus - param_minus)

    # compute learning rate --> see paper (https://arxiv.org/pdf/2106.10055.pdf) Table I (page 6)
    learning_rate = 1  # TODO compute learning rate

    # return new parameters
    return [
        # see paper (https://arxiv.org/pdf/2106.10055.pdf) eq. 7 (page 4)
        param - learning_rate * derivative_numerator / derivative_denominator
        for param, derivative_numerator in zip(param_values, derivative_numerators)
    ]


def construct_circuit() -> Tuple[Circuit, List[float]]:
    # define circuit
    circuit = Circuit(_NUM_QUBITS)

    # set up gates
    # noinspection PyTypeChecker
    last_param_index_by_qubit: List[int] = [None for _ in range(_NUM_QUBITS)]
    num_params = 0

    def _add_parameterized_rotation_to_qubit(qubit_ind: int) -> None:
        nonlocal num_params
        circuit.Ry(Symbol(f"alpha{num_params}"), qubit_ind)
        last_param_index_by_qubit[qubit_ind] = num_params
        num_params += 1

    for i in range(_NUM_QUBITS):
        _add_parameterized_rotation_to_qubit(i)

    for _ in range(_CIRCUIT_BLOCK_DEPTH):
        for i in range(1, _NUM_QUBITS, 2):
            circuit.CX(i - 1, i)
            _add_parameterized_rotation_to_qubit(i - 1)
            _add_parameterized_rotation_to_qubit(i)

        for i in range(2, _NUM_QUBITS, 2):
            circuit.CX(i - 1, i)
            _add_parameterized_rotation_to_qubit(i - 1)
            _add_parameterized_rotation_to_qubit(i)

    circuit.measure_all()

    # initialize gate parameters --> see paper (https://arxiv.org/pdf/2106.10055.pdf) Table I (page 6)
    params = [0 for _ in range(num_params)]
    for last_param_index in last_param_index_by_qubit:
        params[last_param_index] = pi / 2

    return circuit, params


def _output_circuit(circuit: Circuit, params: List[float], file_name: str = "out.html") -> None:
    circuit.symbol_substitution(_get_symbol_dict(params))
    with open(file_name, "w") as file:
        file.write(render_circuit_as_html(circuit))


def _test_circuit_construction():
    # get circuit & params list
    circuit, params = construct_circuit()

    # output circuit
    _output_circuit(circuit, params)


def _test_circuit_training():
    # get circuit & params list
    circuit, params = construct_circuit()

    # compile circuit for backend (here, qiskit-aer = classical simulation)
    compiled = _QUANTUM_BACKEND.get_compiled_circuit(circuit)

    # before energy
    print(f"Before Training: score = {_measure_ansatz(compiled, params, apply_filter=False)}")
    print("---------------------------------------------------")

    # repeated training iterations
    for iteration in range(30):
        # update params
        params = _compute_new_params(compiled, params)

        # print current energy?
        print(f"After Iteration #{iteration}: score = {_measure_ansatz(compiled, params, apply_filter=False)}")

    # after training
    print("---------------------------------------------------")
    print(f"After Training: score = {_measure_ansatz(compiled, params, apply_filter=False)}")


if __name__ == '__main__':
    _test_circuit_training()
