"""QAOA for the hotspot QUBO.

Two phases, on purpose:
  1. TRAIN the QAOA angles on a local simulator (free, unlimited; exact up to ~22 qubits).
  2. SAMPLE the trained circuit once, on the simulator or on real IBM hardware.
     Training on hardware would burn the free QPU minutes in a few runs; one sampling job costs seconds.

Every measured bitstring is decoded into a repair and checked with the same exact checker the
classical solvers use. We report the best valid repair found, and how often QAOA produced valid and
optimal repairs compared with random guessing (the honest yardstick).

Run (simulator):    python src/quantum_solver.py
Run (IBM hardware): python src/quantum_solver.py --backend ibm [--ibm-backend ibm_kingston]
"""
import argparse
import time

import numpy as np
from scipy.optimize import minimize
from qiskit import transpile
from qiskit.circuit.library import qaoa_ansatz

from qubo_builder import bits_from_qiskit, decode

EXACT_LIMIT = 22


def cvar(values, probs, alpha):
    """Average energy of the best `alpha` share of outcomes (CVaR). Works better than the plain
    average for QAOA on optimisation problems."""
    order = np.argsort(values)
    v, p = values[order], probs[order]
    cum = np.cumsum(p)
    take = np.minimum(p, np.maximum(alpha - (cum - p), 0))
    return float((take * v).sum() / alpha)


def make_ansatz(qubo, reps):
    op, _ = qubo.to_ising()
    scale = float(np.max(np.abs(op.coeffs.real)))
    ansatz = qaoa_ansatz(op / scale, reps=reps)
    return ansatz, list(ansatz.parameters)


def train(qubo, reps=3, alpha=0.1, maxiter=300, restarts=3, seed=7):
    from qiskit_aer import AerSimulator

    if qubo.n > EXACT_LIMIT:
        raise ValueError(f"{qubo.n} qubits is too many to train on a laptop; pass trained angles instead")
    ansatz, order = make_ansatz(qubo, reps)
    energies = qubo.all_energies()
    sim = AerSimulator(method="statevector")
    circ = ansatz.copy()
    circ.save_statevector()
    tqc = transpile(circ, sim)

    def objective(theta):
        sv = sim.run(tqc.assign_parameters(dict(zip(order, theta)))).result().get_statevector()
        return cvar(energies, np.abs(np.asarray(sv)) ** 2, alpha)

    gammas = [p for p in order if "γ" in p.name]
    betas = [p for p in order if "β" in p.name]
    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        init = {}
        for k in range(reps):  # "linear ramp" start, a little noise on later restarts
            init[gammas[k]] = (k + 1) / (reps + 1) * 0.8 + rng.normal(0, 0.05 * r)
            init[betas[k]] = (1 - (k + 1) / (reps + 1)) * 0.8 + rng.normal(0, 0.05 * r)
        res = minimize(objective, [init[p] for p in order], method="COBYLA", options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    return list(best.x)


def sample(qubo, angles, reps, shots=4096, backend="aer", ibm_backend=None, ibm_token=None, ibm_instance=None,
           on_submit=None):
    ansatz, order = make_ansatz(qubo, reps)
    circuit = ansatz.assign_parameters(dict(zip(order, angles))).measure_all(inplace=False)
    info = {}
    if backend == "aer":
        from qiskit_aer import AerSimulator
        sim = AerSimulator(method="statevector" if qubo.n <= 26 else "matrix_product_state")
        counts = sim.run(transpile(circuit, sim), shots=shots).result().get_counts()
        info["backend"] = "aer_simulator"
    else:
        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2
        from ibm_quantum import get_service

        service = get_service(ibm_token, ibm_instance)  # API key from the app, .env, or a saved account
        dev = (service.backend(ibm_backend) if ibm_backend
               else service.least_busy(operational=True, simulator=False, min_num_qubits=qubo.n))
        isa = generate_preset_pass_manager(optimization_level=3, backend=dev).run(circuit)
        sampler = SamplerV2(mode=dev)
        sampler.options.dynamical_decoupling.enable = True  # cheap error suppression
        job = sampler.run([isa], shots=shots)
        print(f"Submitted job {job.job_id()} to {dev.name}; waiting in the queue...")
        if on_submit:
            on_submit(job.job_id(), dev.name)
        counts = job.result()[0].data.meas.get_counts()
        ops = isa.count_ops()
        info.update(backend=dev.name, job_id=job.job_id(), depth=isa.depth(),
                    two_qubit_gates=int(sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx"))))
        try:
            info["qpu_seconds"] = job.usage()
        except Exception:
            pass
    return counts, info


def score_counts(spec, sub, model, counts, exact_cost=None):
    shots = sum(counts.values())
    valid, hit, best, best_cost = 0, 0, None, None
    cache = {}
    for b, c in counts.items():
        if b not in cache:
            cache[b] = decode(spec, sub, model, bits_from_qiskit(b))
        choice, cost = cache[b]
        if choice is None:
            continue
        valid += c
        if exact_cost is not None and cost <= exact_cost + 1e-6:
            hit += c
        if best_cost is None or cost < best_cost:
            best, best_cost = choice, cost
    return {"choice": best, "cost": best_cost, "valid_share": round(valid / shots, 4),
            "optimal_share": round(hit / shots, 4) if exact_cost is not None else None}


def random_counts(n, shots, seed=0):
    """The no-quantum baseline: uniformly random bitstrings."""
    rng = np.random.default_rng(seed)
    out = {}
    for row in rng.integers(0, 2, size=(shots, n)):
        b = "".join(str(v) for v in row[::-1])
        out[b] = out.get(b, 0) + 1
    return out


def solve_qaoa(spec, sub, model, exact_cost=None, reps=3, shots=4096, alpha=0.1, backend="aer",
               ibm_backend=None, angles=None, restarts=3, ibm_token=None, ibm_instance=None, on_submit=None):
    q = model["qubo"]
    t0 = time.perf_counter()
    source = "transferred"
    if angles is None:
        angles, source = train(q, reps, alpha, restarts=restarts), "trained"
    t_train = time.perf_counter() - t0
    t1 = time.perf_counter()
    counts, info = sample(q, angles, reps, shots, backend, ibm_backend, ibm_token, ibm_instance, on_submit)
    t_sample = time.perf_counter() - t1
    res = score_counts(spec, sub, model, counts, exact_cost)
    rnd = score_counts(spec, sub, model, random_counts(q.n, shots), exact_cost)
    return {**res, "random_cost": rnd["cost"], "random_valid_share": rnd["valid_share"],
            "random_optimal_share": rnd["optimal_share"], "qubits": q.n, "reps": reps, "shots": shots,
            "angles": [round(float(a), 4) for a in angles], "angles_source": source,
            "train_seconds": round(t_train, 2), "sample_seconds": round(t_sample, 2), **info}


if __name__ == "__main__":
    from common import DATA, RESULTS, apply_disruption, load_json, load_reference, schedule_from_json, summary
    from qubo_builder import tuned_qubo
    from repair import build_subproblem, complete, describe, find_hotspot, shift_right, solve_exact

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["aer", "ibm"], default="aer")
    ap.add_argument("--ibm-backend", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--options", type=int, default=4, help="candidate options per hotspot job")
    a = ap.parse_args()

    spec = apply_disruption(load_reference(), load_json(DATA / "demo_disruption.json"))
    plan = schedule_from_json(load_json(RESULTS / "plan.json"))
    greedy, frozen, _ = shift_right(spec, plan)
    sub = build_subproblem(spec, greedy, frozen, find_hotspot(spec, greedy, frozen), a.options, plan)
    print(describe(spec, sub))
    ex = solve_exact(spec, sub)
    model, check = tuned_qubo(spec, sub, ex["cost"])
    print(f"Exact best cost: {ex['cost']}   QUBO check: {check}")
    res = solve_qaoa(spec, sub, model, ex["cost"], a.reps, a.shots, backend=a.backend, ibm_backend=a.ibm_backend)
    for k, v in res.items():
        if k not in ("choice", "angles"):
            print(f"{k:>22}: {v}")
    if res["choice"]:
        print("QAOA repair, whole night:", summary(spec, complete(spec, sub, res["choice"], plan), plan))
