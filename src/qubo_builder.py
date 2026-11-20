"""Turn a hotspot sub-problem (from repair.build_subproblem) into a QUBO, the input QAOA needs.

The modelling idea, in plain words:
  * One qubit per (job, option): x = 1 means "run this job with this option" (a node + a start time).
  * Each job must take exactly one option -> penalty  A * (sum of that job's x - 1)^2
  * Two options that clash (same node at the same time with too little room, too few licence seats,
    or a successor starting before its predecessor ends) -> penalty  P * x_a * x_b
  * Each option has a cost (lateness against the critical-path deadline, compute $, moves).
    Costs are clipped: an option already hopelessly worse than the job's best one does not need
    its exact size on the chip, and huge numbers would drown out the small differences QAOA must see.

No slack qubits are needed: every rule is either "pick one" or "not both". (In the earlier pilot,
slack qubits were what made QAOA struggle.)

Run:  python src/qubo_builder.py   -> builds the demo hotspot QUBO and checks it by brute force
"""
from itertools import combinations

import numpy as np

from repair import choice_cost, choice_feasible

CLIP = 60.0  # largest cost difference between two options of one job that the QUBO keeps


class QUBO:
    """Energy(x) = offset + sum_i linear[i] x_i + sum_{i<j} quad[i,j] x_i x_j"""

    def __init__(self):
        self.names, self.linear, self.quad, self.offset = [], [], {}, 0.0

    def var(self, name):
        self.names.append(name)
        self.linear.append(0.0)
        return len(self.names) - 1

    def add_quad(self, i, j, c):
        if i == j:
            self.linear[i] += c
            return
        key = (min(i, j), max(i, j))
        self.quad[key] = self.quad.get(key, 0.0) + c

    def add_one_hot(self, weight, idx):
        """weight * (sum x - 1)^2 = weight * (1 - sum x + 2 * sum_{pairs} x_i x_j)"""
        self.offset += weight
        for i in idx:
            self.linear[i] -= weight
        for i, j in combinations(idx, 2):
            self.add_quad(i, j, 2 * weight)

    @property
    def n(self):
        return len(self.names)

    def energy(self, bits):
        e = self.offset + sum(c * b for c, b in zip(self.linear, bits))
        return e + sum(c * bits[i] * bits[j] for (i, j), c in self.quad.items())

    def all_energies(self, chunk=1 << 18):
        """Energy of all 2^n bitstrings (n <= ~24). Index bit i = variable i."""
        n, lin = self.n, np.array(self.linear)
        qi = np.array([k[0] for k in self.quad], dtype=np.int64)
        qj = np.array([k[1] for k in self.quad], dtype=np.int64)
        qc = np.array(list(self.quad.values()))
        out = np.empty(1 << n)
        shifts = np.arange(n, dtype=np.int64)
        for start in range(0, 1 << n, chunk):
            idx = np.arange(start, min(start + chunk, 1 << n), dtype=np.int64)
            X = ((idx[:, None] >> shifts) & 1).astype(np.float64)
            e = self.offset + X @ lin
            if len(qc):
                e += (X[:, qi] * X[:, qj]) @ qc
            out[start:start + len(idx)] = e
        return out

    def to_ising(self):
        """Substitute x = (1 - Z)/2. Returns (SparsePauliOp, constant). Qubit i = variable i."""
        from qiskit.quantum_info import SparsePauliOp

        const, z, zz = self.offset, {}, {}
        for i, c in enumerate(self.linear):
            const += c / 2
            z[i] = z.get(i, 0.0) - c / 2
        for (i, j), c in self.quad.items():
            const += c / 4
            z[i] = z.get(i, 0.0) - c / 4
            z[j] = z.get(j, 0.0) - c / 4
            zz[(i, j)] = zz.get((i, j), 0.0) + c / 4
        terms = [("Z", [i], c) for i, c in z.items() if abs(c) > 1e-12]
        terms += [("ZZ", [i, j], c) for (i, j), c in zz.items() if abs(c) > 1e-12]
        return SparsePauliOp.from_sparse_list(terms, num_qubits=self.n), const


def build_qubo(sub, penalty=None):
    q = QUBO()
    index = {}
    for j in sub["jobs"]:
        best = min(o["cost"] for o in sub["options"][j])
        for k, o in enumerate(sub["options"][j]):
            i = q.var(f"x[{j},{o['node']}@{k}]")
            index[(j, k)] = i
            q.linear[i] += min(o["cost"] - best, CLIP)
        q.offset += best
    if penalty is None:
        penalty = CLIP + 1.0
    for j in sub["jobs"]:
        q.add_one_hot(penalty, [index[(j, k)] for k in range(len(sub["options"][j]))])
    for c in sub["conflicts"]:
        q.add_quad(index[tuple(c["a"])], index[tuple(c["b"])], penalty)
    return {"qubo": q, "index": index, "penalty": penalty}


def decode(spec, sub, model, bits):
    """bits -> (choice or None, true cost or None). None = not a valid repair."""
    choice = {}
    for j in sub["jobs"]:
        on = [k for k in range(len(sub["options"][j])) if bits[model["index"][(j, k)]]]
        if len(on) != 1:
            return None, None
        choice[j] = on[0]
    if not choice_feasible(spec, sub, choice):
        return None, None
    return choice, choice_cost(sub, choice)


def bits_from_index(k, n):
    return [(k >> i) & 1 for i in range(n)]


def bits_from_qiskit(bitstring):
    """Qiskit prints qubit 0 as the RIGHTMOST character, so reverse it."""
    return [int(c) for c in reversed(bitstring)]


def tuned_qubo(spec, sub, exact_cost, max_brute=22):
    """Build the QUBO and, when small enough to check by brute force, raise the penalty until the
    QUBO's lowest-energy bitstring is a valid repair with the exact optimal cost."""
    model = build_qubo(sub)
    if model["qubo"].n > max_brute:
        return model, {"checked": False}
    for _ in range(6):
        energies = model["qubo"].all_energies()
        k = int(np.argmin(energies))
        choice, cost = decode(spec, sub, model, bits_from_index(k, model["qubo"].n))
        if choice is not None and exact_cost is not None and cost <= exact_cost + 1e-6:
            return model, {"checked": True, "matches_exact": True, "penalty": model["penalty"]}
        if choice is not None:
            # valid but not optimal: costs were clipped; report it rather than loop forever
            return model, {"checked": True, "matches_exact": False, "penalty": model["penalty"],
                           "qubo_min_cost": cost}
        model = build_qubo(sub, penalty=model["penalty"] * 2)
    return model, {"checked": True, "matches_exact": False, "penalty": model["penalty"]}


if __name__ == "__main__":
    from common import DATA, RESULTS, apply_disruption, load_json, load_reference, schedule_from_json
    from repair import build_subproblem, find_hotspot, shift_right, solve_exact

    spec = apply_disruption(load_reference(), load_json(DATA / "demo_disruption.json"))
    plan = schedule_from_json(load_json(RESULTS / "plan.json"))
    greedy, frozen, _ = shift_right(spec, plan)
    sub = build_subproblem(spec, greedy, frozen, find_hotspot(spec, greedy, frozen), options_per_job=4, plan=plan)
    ex = solve_exact(spec, sub)
    model, check = tuned_qubo(spec, sub, ex["cost"])
    q = model["qubo"]
    op, const = q.to_ising()
    print(f"Qubits: {q.n} ({len(sub['jobs'])} jobs x options), {len(q.quad)} pairwise terms, {len(op)} Pauli terms")
    print("Variables:", ", ".join(q.names))
    print(f"Exact best cost {ex['cost']} from {ex['combinations']} combinations; brute-force check: {check}")
