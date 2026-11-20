"""Classical planner: schedule the whole night exactly with Google OR-Tools CP-SAT.

Two uses:
  plan_night(spec)                      -> tonight's plan, made before the batch starts
  plan_night(spec, plan=..., frozen=...) -> full re-plan after a disruption: jobs already
                                           started stay put, every other job may move
                                           (moves are penalised so the plan stays familiar)

Run:  python src/planner.py              -> plans the reference night, saves results/plan.json
"""
import time

from ortools.sat.python import cp_model

from common import (HORIZON, RESULTS, duration, eligible_nodes, job_cost, job_map, load_reference, node_map,
                    release, save_json, schedule_table, schedule_to_json, summary, to_hhmm, validate_spec)

SCALE = 100  # CP-SAT works in whole numbers: costs are multiplied by 100


def _merged(blackouts):
    """Overlapping blackout windows must be merged, or together they would 'use' the node twice."""
    out = []
    for b0, b1, _ in sorted(blackouts):
        if b1 <= b0:
            continue
        if out and b0 <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b1)
        else:
            out.append([b0, b1])
    return out


def plan_night(spec, plan=None, frozen=None, time_limit_s=30, workers=8, seed=0):
    jobs, nodes = job_map(spec), node_map(spec)
    frozen = frozen or {}
    w = spec["weights"]
    m = cp_model.CpModel()
    start, end, pick, intervals = {}, {}, {}, []
    terms = []

    for j, job in jobs.items():
        lo = release(spec, job)
        if j in frozen:
            fn, fs = frozen[j]
            start[j] = m.NewConstant(fs)
            end[j] = m.NewConstant(fs + duration(spec, job, nodes[fn]))
            options = [nodes[fn]]
        else:
            lo = max(lo, spec.get("as_of", 0))
            start[j] = m.NewIntVar(lo, HORIZON + 300, f"s_{j}")
            end[j] = m.NewIntVar(lo, HORIZON + 600, f"e_{j}")
            options = eligible_nodes(spec, job)
        for node in options:
            n = node["id"]
            d = duration(spec, job, node)
            p = m.NewConstant(1) if j in frozen else m.NewBoolVar(f"p_{j}_{n}")
            pick[(j, n)] = p
            iv = m.NewOptionalIntervalVar(start[j], d, end[j], p, f"iv_{j}_{n}")
            intervals.append((j, n, iv))
            if j not in frozen:
                # node-dependent part of the cost (compute $)
                terms.append(int(round(SCALE * w["compute_cost"] * node["cost_per_hour"] * d / 60)) * p)
                if plan and j in plan and plan[j][0] != n:
                    terms.append(int(round(SCALE * w["move_node"])) * p)
        if j not in frozen:
            m.AddExactlyOne([pick[(j, nd["id"])] for nd in options])
            late = m.NewIntVar(0, HORIZON + 600, f"late_{j}")
            m.Add(late >= end[j] - job["deadline"])
            wl = w["priority"][job["priority"]] * (w["hard_multiplier"] if job["hard"] else 1.0)
            terms.append(int(round(SCALE * wl)) * late)
            terms.append(int(round(SCALE * w["finish_early"])) * end[j])
            if plan and j in plan:
                shift = m.NewIntVar(0, HORIZON + 600, f"shift_{j}")
                m.AddAbsEquality(shift, start[j] - plan[j][1])
                terms.append(int(round(SCALE * w["move_minutes"])) * shift)

    # capacity on every node (cores, memory, GPUs), with blackout windows using the whole node
    for node in spec["nodes"]:
        n = node["id"]
        mine = [(j, iv) for (j, nn, iv) in intervals if nn == n]
        blk = [m.NewIntervalVar(b0, b1 - b0, b1, f"blk_{n}_{i}") for i, (b0, b1) in enumerate(_merged(node["blackouts"]))]
        for res in ("cores", "mem", "gpu"):
            if node[res] == 0:
                continue
            m.AddCumulative([iv for _, iv in mine] + blk, [jobs[j][res] for j, _ in mine] + [node[res]] * len(blk), node[res])
    # licences are shared across all nodes
    for lid, lic in spec["licenses"].items():
        ivs = [iv for (j, n, iv) in intervals if jobs[j]["license"] == lid]
        if ivs:
            m.AddCumulative(ivs, [1] * len(ivs), lic["seats"])
    # dependencies
    for e in spec["edges"]:
        if e["pred"] in jobs and e["succ"] in jobs:
            m.Add(start[e["succ"]] >= end[e["pred"]] + e["lag"])

    m.Minimize(sum(terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s * 4  # safety net on slow laptops
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.interleave_search = True  # deterministic: the same plan on every run and every laptop
    solver.parameters.max_deterministic_time = time_limit_s
    t0 = time.perf_counter()
    status = solver.Solve(m)
    secs = time.perf_counter() - t0
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": "infeasible", "schedule": {}, "seconds": round(secs, 2)}
    schedule = {}
    for (j, n), p in pick.items():
        if solver.Value(p):
            schedule[j] = (n, solver.Value(start[j]))
    return {"status": "optimal" if status == cp_model.OPTIMAL else "good (time limit reached)",
            "schedule": schedule, "seconds": round(secs, 2)}


if __name__ == "__main__":
    spec = load_reference()
    errs = validate_spec(spec)
    if errs:
        raise SystemExit("\n".join(errs))
    res = plan_night(spec)
    print(f"Status: {res['status']} in {res['seconds']}s")
    print(summary(spec, res["schedule"]))
    print(schedule_table(spec, res["schedule"]))
    save_json(schedule_to_json(res["schedule"]), RESULTS / "plan.json")
    print(f"Saved {RESULTS / 'plan.json'}")
