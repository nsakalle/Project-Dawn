"""Run every benchmark instance through: shift-right, exact hotspot solve, QAOA hotspot solve,
random guessing, and the full CP-SAT re-plan. One row per instance in results/benchmark.csv.

Run (simulator):  python src/benchmark.py
Run (hardware):   python src/benchmark.py --backend ibm --tiers small,medium,large
Above --train-limit qubits, QAOA reuses the median angles trained on smaller instances.
"""
import argparse
import csv
import glob

import numpy as np

from common import DATA, RESULTS, apply_disruption, load_json, load_reference, schedule_from_json, summary
from planner import plan_night
from quantum_solver import solve_qaoa
from qubo_builder import tuned_qubo
from repair import build_subproblem, choice_cost, complete, find_hotspot, shift_right, solve_exact

COLUMNS = ["instance", "tier", "qubits", "hotspot_jobs", "conflicts",
           "shift_right_sub_cost", "exact_sub_cost", "qaoa_sub_cost", "qaoa_matches_exact",
           "qaoa_valid_share", "random_valid_share", "qaoa_optimal_share", "random_optimal_share",
           "night_shift_right", "night_hotspot_exact", "night_hotspot_qaoa", "night_full_cpsat",
           "hard_misses_shift_right", "hard_misses_qaoa", "hard_misses_cpsat",
           "qubo_check", "angles_source", "train_seconds", "backend", "job_id", "qpu_seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="toy,small,medium,large")
    ap.add_argument("--backend", choices=["aer", "ibm"], default="aer")
    ap.add_argument("--ibm-backend", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--train-limit", type=int, default=18)
    ap.add_argument("--out", default=str(RESULTS / "benchmark.csv"))
    a = ap.parse_args()

    base = load_reference()
    plan = schedule_from_json(load_json(RESULTS / "plan.json"))
    rows, learned = [], []
    for tier in a.tiers.split(","):
        for path in sorted(glob.glob(str(DATA / "instances" / f"{tier}-*.json"))):
            inst = load_json(path)
            name = path.split("/")[-1][:-5]
            spec = apply_disruption(base, inst["disruption"])
            greedy, frozen, _ = shift_right(spec, plan)
            hs = find_hotspot(spec, greedy, frozen, max_jobs=inst["max_jobs"], seed_rank=inst.get("seed_rank", 0))
            sub = build_subproblem(spec, greedy, frozen, hs, inst["options"], plan)
            n = sum(len(v) for v in sub["options"].values())
            if a.backend == "aer" and n > 26:
                print(f"{name}: {n} qubits is too big for a laptop simulator; run it with --backend ibm")
                continue
            ex = solve_exact(spec, sub)
            greedy_choice = {j: next(i for i, o in enumerate(sub["options"][j])
                                     if (o["node"], o["start"]) == tuple(greedy[j])) for j in sub["jobs"]}
            model, check = tuned_qubo(spec, sub, ex["cost"])
            angles = None
            if n > a.train_limit:
                if not learned:
                    print(f"{name}: no trained angles to transfer yet; skipping")
                    continue
                angles = list(np.median(np.array(learned), axis=0))
            q = solve_qaoa(spec, sub, model, ex["cost"], a.reps, a.shots, backend=a.backend,
                           ibm_backend=a.ibm_backend, angles=angles)
            if q["angles_source"] == "trained":
                learned.append(q["angles"])
            night_g = summary(spec, greedy, plan)
            night_e = summary(spec, complete(spec, sub, ex["choice"], plan), plan)
            night_q = summary(spec, complete(spec, sub, q["choice"], plan), plan) if q["choice"] else None
            full = plan_night(spec, plan=plan, frozen=frozen, time_limit_s=10)
            night_c = summary(spec, full["schedule"], plan)
            row = {
                "instance": name, "tier": inst["tier"], "qubits": n, "hotspot_jobs": len(hs),
                "conflicts": len(sub["conflicts"]),
                "shift_right_sub_cost": choice_cost(sub, greedy_choice), "exact_sub_cost": ex["cost"],
                "qaoa_sub_cost": q["cost"],
                "qaoa_matches_exact": q["cost"] is not None and q["cost"] <= ex["cost"] + 1e-6,
                "qaoa_valid_share": q["valid_share"], "random_valid_share": q["random_valid_share"],
                "qaoa_optimal_share": q["optimal_share"], "random_optimal_share": q["random_optimal_share"],
                "night_shift_right": night_g["score"], "night_hotspot_exact": night_e["score"],
                "night_hotspot_qaoa": night_q["score"] if night_q else None, "night_full_cpsat": night_c["score"],
                "hard_misses_shift_right": night_g["hard_misses"],
                "hard_misses_qaoa": night_q["hard_misses"] if night_q else None,
                "hard_misses_cpsat": night_c["hard_misses"],
                "qubo_check": "matches" if check.get("matches_exact") else ("not checked" if not check.get("checked") else "differs"),
                "angles_source": q["angles_source"], "train_seconds": q["train_seconds"],
                "backend": q.get("backend"), "job_id": q.get("job_id", ""), "qpu_seconds": q.get("qpu_seconds", ""),
            }
            rows.append(row)
            _write(a.out, rows)  # save after every instance, so a long run can be stopped safely
            print(f"{name}: {n} qubits, exact {ex['cost']}, QAOA {q['cost']}, valid {q['valid_share']:.1%} "
                  f"(random {q['random_valid_share']:.1%}), night {night_g['score']} -> {row['night_hotspot_qaoa']} "
                  f"(CP-SAT {night_c['score']})", flush=True)
    _write(a.out, rows)
    print(f"Wrote {len(rows)} rows to {a.out}")


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
