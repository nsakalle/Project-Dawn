"""Make random disruptions of the reference night, keep the ones that produce a hotspot, and sort
them into tiers by qubit count. Each instance file holds the disruption JSON plus the hotspot size
settings, so benchmark.py can rebuild exactly the same sub-problem.

Run:  python src/generate_instances.py      -> data/instances/<tier>-<n>.json
"""
import json
import random

from common import (DATA, RESULTS, apply_disruption, load_json, load_reference, save_json, schedule_from_json,
                    to_hhmm)
from repair import build_subproblem, find_hotspot, shift_right

TIERS = {"toy": (4, 8), "small": (9, 16), "medium": (17, 24), "large": (25, 40)}
PER_TIER = 5
OUTAGE_NODES = ["HMEM-01", "HMEM-02", "GPU-01", "GPU-02", "DB-01", "GEN-01", "GEN-03"]


def random_disruption(spec, rng):
    as_of = rng.choice(range(120, 300, 15))  # 22:00 to 00:45
    d = {"as_of": to_hhmm(as_of), "external_delays": [], "node_outages": [], "deadline_changes": [],
         "priority_changes": [], "job_holds": [], "use_p90_runtimes": False, "assumptions": ["synthetic"]}
    later = [k for k, e in spec["external"].items() if e["expected"] > as_of]
    for k in rng.sample(later, min(len(later), rng.choice([1, 1, 2]))):
        d["external_delays"].append({"ext_id": k, "new_time": to_hhmm(spec["external"][k]["expected"] + rng.choice([30, 60, 90, 120, 150]))})
    if rng.random() < 0.85:
        start = as_of + rng.choice([15, 30, 60, 90, 120])
        d["node_outages"].append({"node_id": rng.choice(OUTAGE_NODES), "start": to_hhmm(start),
                                  "end": to_hhmm(min(start + rng.choice([60, 120, 180, 240]), 700)), "reason": "synthetic"})
    if rng.random() < 0.4:
        j = rng.choice([j for j in spec["jobs"] if j["priority"] in ("P1", "P2")])
        d["deadline_changes"].append({"job_id": j["id"], "deadline": to_hhmm(max(j["deadline"] - rng.choice([30, 60]), as_of + 60))})
    return d


def generate(seed=1):
    base = load_reference()
    plan = schedule_from_json(load_json(RESULTS / "plan.json"))
    rng = random.Random(seed)
    counts = {t: 0 for t in TIERS}
    seen = set()
    tries = 0
    while any(c < PER_TIER for c in counts.values()) and tries < 20000:
        tries += 1
        d = random_disruption(base, rng)
        max_jobs, options = rng.choice([2, 3, 4, 5, 6, 7, 8]), rng.choice([2, 3, 4, 5])
        seed_rank = rng.choice([0, 0, 1, 2])
        spec = apply_disruption(base, d)
        try:
            greedy, frozen, _ = shift_right(spec, plan)
        except RuntimeError:
            continue
        hs = find_hotspot(spec, greedy, frozen, max_jobs=max_jobs, seed_rank=seed_rank)
        if len(hs) < 2:
            continue
        sub = build_subproblem(spec, greedy, frozen, hs, options, plan)
        n = sum(len(v) for v in sub["options"].values())
        key = (tuple(sorted(hs)), n)
        if key in seen:
            continue
        tier = next((t for t, (lo, hi) in TIERS.items() if lo <= n <= hi), None)
        if tier is None or counts[tier] >= PER_TIER or not sub["conflicts"]:
            continue
        seen.add(key)
        counts[tier] += 1
        save_json({"disruption": d, "max_jobs": max_jobs, "options": options, "seed_rank": seed_rank,
                   "tier": tier, "qubits": n},
                  DATA / "instances" / f"{tier}-{counts[tier]}.json")
        print(f"{tier}-{counts[tier]}: {n} qubits, hotspot {hs}, {len(sub['conflicts'])} conflicts")
    print(f"done after {tries} random disruptions: {counts}")


if __name__ == "__main__":
    generate()
