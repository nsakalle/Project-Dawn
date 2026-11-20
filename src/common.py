"""Shared building blocks: load the CSVs, apply tonight's disruption, check and score a schedule.

Every solver in the pilot (classical planner, greedy repair, exact repair, QAOA repair) is judged
with the SAME checker (`violations`) and the SAME score (`schedule_score`), so results compare fairly.

Time is counted in minutes from the start of the batch window (20:00 = 0, 08:00 = 720).
A schedule is a dict {job_id: (node_id, start_minute)}.
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

WINDOW_START = 20 * 60          # 20:00
HORIZON = 900                   # plan up to 11:00; anything after 08:00 counts as late anyway
STEP = 5                        # durations are rounded up to 5-minute steps

WEIGHTS = {
    "priority": {"P1": 10.0, "P2": 5.0, "P3": 2.0, "P4": 1.0},  # cost per minute late
    "hard_multiplier": 5.0,        # a hard deadline missed costs 5x a soft one
    "compute_cost": 1.0,           # per USD of node time
    "finish_early": 0.01,          # per minute of finish time: prefer running jobs early
    "move_node": 3.0,              # repair only: moving a job to another node
    "move_minutes": 0.02,          # repair only: per minute a job's start is shifted
}


# ------------------------------------------------------------------ time helpers
def to_min(hhmm):
    """'21:30' -> 90, '02:00' -> 360 (minutes after 20:00; hours before noon are 'tomorrow')."""
    h, m = map(int, hhmm.split(":"))
    t = h * 60 + m
    if h < 12:
        t += 24 * 60
    return t - WINDOW_START


def to_hhmm(m):
    t = (WINDOW_START + int(m)) % (24 * 60)
    return f"{t // 60:02d}:{t % 60:02d}"


# ------------------------------------------------------------------ loading
def _rows(name, data_dir):
    with open(Path(data_dir) / name, newline="") as f:
        return list(csv.DictReader(f))


def _split(v):
    return [x.strip() for x in (v or "").split(";") if x.strip()]


def load_reference(data_dir=DATA):
    """Read the 7 CSV files into one plain dictionary (the 'spec')."""
    nodes = []
    for r in _rows("resources.csv", data_dir):
        nodes.append({
            "id": r["node_id"], "name": r["node_name"], "pool": r["pool"], "datacenter": r["datacenter"],
            "cores": int(r["cpu_cores"]), "mem": int(r["memory_gb"]), "gpu": int(r["gpu_count"]),
            "speed": float(r["speed_factor"]), "cost_per_hour": float(r["cost_per_hour_usd"]),
            "blackouts": [],
        })
        a, b = to_min(r["available_from"]), to_min(r["available_to"])
        if a > 0:
            nodes[-1]["blackouts"].append([0, a, "not available yet"])
        if b < 720:  # leaves before 08:00; after 08:00 the batch may overrun (late, but it still runs)
            nodes[-1]["blackouts"].append([b, HORIZON, "not available"])
    by_node = {n["id"]: n for n in nodes}
    for r in _rows("maintenance_windows.csv", data_dir):
        by_node[r["node_id"]]["blackouts"].append([to_min(r["start"]), to_min(r["end"]), r["reason"]])

    external = {r["ext_id"]: {"description": r["description"], "source": r["source_system"], "type": r["ext_type"],
                              "expected": to_min(r["expected_time"]), "latest": to_min(r["latest_time"]),
                              "owner": r["owner_team"]}
                for r in _rows("external_dependencies.csv", data_dir)}
    licenses = {r["license_id"]: {"product": r["product"], "seats": int(r["concurrent_seats"])}
                for r in _rows("licenses.csv", data_dir)}
    jobs = []
    for r in _rows("jobs.csv", data_dir):
        jobs.append({
            "id": r["job_id"], "name": r["job_name"], "app": r["application"], "team": r["owner_team"],
            "type": r["job_type"], "priority": r["priority"],
            "cores": int(r["cpu_cores"]), "mem": int(r["memory_gb"]), "gpu": int(r["gpu_count"]),
            "runtime": int(r["est_runtime_min"]), "runtime_p90": int(r["runtime_p90_min"]),
            "pools": _split(r["allowed_pools"]), "license": r["license_id"] or None,
            "earliest": to_min(r["earliest_start"]), "deadline": to_min(r["deadline"]),
            "hard": r["deadline_type"] == "hard", "ext": _split(r["external_deps"]),
        })
    edges = [{"pred": r["predecessor_id"], "succ": r["successor_id"], "lag": int(r["lag_min"])}
             for r in _rows("job_dependencies.csv", data_dir)]
    return {"jobs": jobs, "edges": edges, "nodes": nodes, "licenses": licenses, "external": external,
            "weights": json.loads(json.dumps(WEIGHTS)), "as_of": 0, "held": [], "notes": [],
            "use_p90": False}


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=list)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ disruption
def apply_disruption(spec, d):
    """Return a copy of spec with tonight's changes applied. `d` is the JSON from watsonx (or by hand):
    {as_of, external_delays[{ext_id, new_time}], node_outages[{node_id, start, end, reason}],
     deadline_changes[{job_id, deadline, deadline_type?}], priority_changes[{job_id, priority}],
     job_holds[job_id], use_p90_runtimes, assumptions[]}"""
    s = json.loads(json.dumps(spec))
    s["as_of"] = to_min(d["as_of"]) if d.get("as_of") else 0
    jobs = {j["id"]: j for j in s["jobs"]}
    nodes = {n["id"]: n for n in s["nodes"]}
    for e in d.get("external_delays", []):
        if e["ext_id"] in s["external"]:
            s["external"][e["ext_id"]]["expected"] = to_min(e["new_time"])
            s["external"][e["ext_id"]]["latest"] = max(s["external"][e["ext_id"]]["latest"], to_min(e["new_time"]))
    for o in d.get("node_outages", []):
        if o["node_id"] in nodes:
            nodes[o["node_id"]]["blackouts"].append([to_min(o["start"]), to_min(o["end"]), o.get("reason", "outage")])
    for c in d.get("deadline_changes", []):
        if c["job_id"] in jobs:
            jobs[c["job_id"]]["deadline"] = to_min(c["deadline"])
            if c.get("deadline_type"):
                jobs[c["job_id"]]["hard"] = c["deadline_type"] == "hard"
    for c in d.get("priority_changes", []):
        if c["job_id"] in jobs:
            jobs[c["job_id"]]["priority"] = c["priority"]
    held = set(h for h in d.get("job_holds", []) if h in jobs)
    # holding a job also holds everything downstream of it
    changed = True
    while changed:
        changed = False
        for e in s["edges"]:
            if e["pred"] in held and e["succ"] not in held:
                held.add(e["succ"])
                s["notes"].append(f"{e['succ']} also held because it depends on {e['pred']}")
                changed = True
    s["held"] = sorted(held)
    s["use_p90"] = bool(d.get("use_p90_runtimes"))
    s["notes"] += d.get("assumptions", [])
    return s


# ------------------------------------------------------------------ job facts
def job_map(spec):
    return {j["id"]: j for j in spec["jobs"] if j["id"] not in spec.get("held", [])}


def node_map(spec):
    return {n["id"]: n for n in spec["nodes"]}


def release(spec, job):
    """Earliest start: the job's own earliest time and every external event it waits for."""
    t = job["earliest"]
    for e in job["ext"]:
        t = max(t, spec["external"][e]["expected"])
    return t


def duration(spec, job, node):
    base = job["runtime_p90"] if spec.get("use_p90") else job["runtime"]
    return int(math.ceil(base * node["speed"] / STEP) * STEP)


def eligible(job, node):
    return (node["pool"] in job["pools"] and job["cores"] <= node["cores"]
            and job["mem"] <= node["mem"] and job["gpu"] <= node["gpu"])


def eligible_nodes(spec, job):
    return [n for n in spec["nodes"] if eligible(job, n)]


def preds_of(spec):
    p = defaultdict(list)
    for e in spec["edges"]:
        p[e["succ"]].append((e["pred"], e["lag"]))
    return p


def succs_of(spec):
    s = defaultdict(list)
    for e in spec["edges"]:
        s[e["pred"]].append((e["succ"], e["lag"]))
    return s


def topo_order(spec, ids=None):
    jobs = job_map(spec)
    ids = set(ids) if ids is not None else set(jobs)
    indeg = {j: 0 for j in ids}
    for e in spec["edges"]:
        if e["pred"] in ids and e["succ"] in ids:
            indeg[e["succ"]] += 1
    order, ready = [], sorted(j for j, d in indeg.items() if d == 0)
    sv = succs_of(spec)
    while ready:
        j = ready.pop(0)
        order.append(j)
        for s, _ in sv[j]:
            if s in indeg:
                indeg[s] -= 1
                if indeg[s] == 0:
                    ready.append(s)
                    ready.sort()
    return order


# ------------------------------------------------------------------ validation
def validate_spec(spec):
    """Problems that make the night impossible to plan. Empty list = OK."""
    errs = []
    jobs = {j["id"]: j for j in spec["jobs"]}
    for e in spec["edges"]:
        for k in ("pred", "succ"):
            if e[k] not in jobs:
                errs.append(f"Dependency refers to unknown job {e[k]}.")
    if len(topo_order(spec)) != len(job_map(spec)):
        errs.append("The dependencies contain a cycle.")
    for j in job_map(spec).values():
        if not eligible_nodes(spec, j):
            errs.append(f"{j['id']} ({j['cores']} cores, {j['mem']} GB, {j['gpu']} GPU, pools {j['pools']}) fits no node.")
        for e in j["ext"]:
            if e not in spec["external"]:
                errs.append(f"{j['id']} waits for unknown external event {e}.")
        if j["license"] and j["license"] not in spec["licenses"]:
            errs.append(f"{j['id']} needs unknown licence {j['license']}.")
    return errs


# ------------------------------------------------------------------ checking and scoring
def _overlap(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1


def violations(spec, schedule):
    """Every hard-rule break in a schedule. Empty list = the schedule can actually run."""
    out = []
    jobs, nodes, preds = job_map(spec), node_map(spec), preds_of(spec)
    for j in jobs:
        if j not in schedule:
            out.append(f"{j} is not scheduled.")
    end = {}
    for j, (n, s) in schedule.items():
        if j not in jobs:
            continue
        job, node = jobs[j], nodes[n]
        d = duration(spec, job, node)
        end[j] = s + d
        if not eligible(job, node):
            out.append(f"{j} cannot run on {n}.")
        if s < release(spec, job):
            out.append(f"{j} starts {to_hhmm(s)}, before its inputs are ready at {to_hhmm(release(spec, job))}.")
        for b0, b1, why in node["blackouts"]:
            if _overlap(s, s + d, b0, b1):
                out.append(f"{j} on {n} overlaps {why} ({to_hhmm(b0)}-{to_hhmm(b1)}).")
    for j, (n, s) in schedule.items():
        for p, lag in preds.get(j, []):
            if p in end and end[p] + lag > s:
                out.append(f"{j} starts before {p} has finished.")
    # capacity, checked at every moment a job starts
    by_node = defaultdict(list)
    lic = defaultdict(list)
    for j, (n, s) in schedule.items():
        if j in jobs:
            by_node[n].append((s, end[j], jobs[j]))
            if jobs[j]["license"]:
                lic[jobs[j]["license"]].append((s, end[j], j))
    for n, items in by_node.items():
        node = nodes[n]
        for t in sorted({s for s, _, _ in items}):
            live = [job for s, e, job in items if s <= t < e]
            for res in ("cores", "mem", "gpu"):
                used = sum(job[res] for job in live)
                if used > node[res]:
                    out.append(f"{n} over {res} at {to_hhmm(t)}: {used} > {node[res]} ({', '.join(x['id'] for x in live)}).")
    for l, items in lic.items():
        for t in sorted({s for s, _, _ in items}):
            live = [j for s, e, j in items if s <= t < e]
            if len(live) > spec["licenses"][l]["seats"]:
                out.append(f"{l} over its {spec['licenses'][l]['seats']} seats at {to_hhmm(t)} ({', '.join(live)}).")
    return out


def job_cost(spec, job, node, start, plan_entry=None):
    """Cost of one job placed on one node at one start time (lower is better)."""
    w = spec["weights"]
    d = duration(spec, job, node)
    end = start + d
    late = max(0, end - job["deadline"])
    c = w["priority"][job["priority"]] * (w["hard_multiplier"] if job["hard"] else 1.0) * late
    c += w["compute_cost"] * node["cost_per_hour"] * d / 60.0
    c += w["finish_early"] * end
    if plan_entry is not None:
        pn, ps = plan_entry
        c += w["move_node"] * (pn != node["id"]) + w["move_minutes"] * abs(start - ps)
    return c


def schedule_score(spec, schedule, plan=None):
    jobs, nodes = job_map(spec), node_map(spec)
    total = 0.0
    for j, (n, s) in schedule.items():
        if j in jobs:
            total += job_cost(spec, jobs[j], nodes[n], s, plan.get(j) if plan else None)
    return round(total, 2)


def sla_report(spec, schedule):
    """Deadline misses: list of (job_id, minutes_late, hard?)."""
    jobs, nodes = job_map(spec), node_map(spec)
    out = []
    for j, (n, s) in sorted(schedule.items()):
        if j in jobs:
            late = s + duration(spec, jobs[j], nodes[n]) - jobs[j]["deadline"]
            if late > 0:
                out.append((j, late, jobs[j]["hard"]))
    return out


def summary(spec, schedule, plan=None):
    misses = sla_report(spec, schedule)
    jobs, nodes = job_map(spec), node_map(spec)
    moved = [j for j in schedule if plan and j in plan and tuple(plan[j]) != tuple(schedule[j])]
    usd = sum(nodes[n]["cost_per_hour"] * duration(spec, jobs[j], nodes[n]) / 60 for j, (n, s) in schedule.items() if j in jobs)
    finish = max(s + duration(spec, jobs[j], nodes[n]) for j, (n, s) in schedule.items() if j in jobs)
    return {
        "score": schedule_score(spec, schedule, plan),
        "violations": len(violations(spec, schedule)),
        "sla_misses": len(misses),
        "hard_misses": sum(1 for _, _, h in misses if h),
        "minutes_late": sum(m for _, m, _ in misses),
        "compute_usd": round(usd, 2),
        "batch_finishes": to_hhmm(finish),
        "jobs_moved": len(moved),
    }


def schedule_table(spec, schedule, plan=None, only=None):
    jobs, nodes = job_map(spec), node_map(spec)
    rows = [f"{'Job':<7}{'Name':<34}{'Node':<9}{'Start':<7}{'End':<7}{'Deadline':<10}{'Status'}"]
    for j, (n, s) in sorted(schedule.items(), key=lambda kv: (kv[1][1], kv[0])):
        if j not in jobs or (only and j not in only):
            continue
        job = jobs[j]
        e = s + duration(spec, job, nodes[n])
        status = "on time" if e <= job["deadline"] else f"LATE {e - job['deadline']} min" + (" (HARD)" if job["hard"] else "")
        if plan and j in plan and tuple(plan[j]) != (n, s):
            pn, ps = plan[j]
            status += f"  [moved from {pn} {to_hhmm(ps)}]"
        rows.append(f"{j:<7}{job['name'][:32]:<34}{n:<9}{to_hhmm(s):<7}{to_hhmm(e):<7}{to_hhmm(job['deadline']):<10}{status}")
    return "\n".join(rows)


def schedule_to_json(schedule):
    return {j: [n, int(s)] for j, (n, s) in schedule.items()}


def schedule_from_json(obj):
    return {j: (v[0], int(v[1])) for j, v in obj.items()}
