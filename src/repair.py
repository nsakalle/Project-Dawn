"""Repairing tonight's plan after a disruption (late feed, node outage, changed deadline...).

Three ways to repair, all scored with the same function so they compare fairly:

1. shift_right()   What a typical rule-based batch scheduler does today: keep the planned order and
                   node, push each job later until it can run. Fast, familiar, often leaves SLAs late.
2. hotspot repair  Find the few jobs fighting over a scarce resource (high-memory nodes, SAS seats,
                   GPUs) around the disruption, give each a handful of candidate (node, start) options,
                   and pick the best combination. THIS combination problem is what QAOA solves.
                   Everything downstream is then re-dispatched.
3. planner.plan_night(..., frozen=...)  Full classical re-plan with CP-SAT: the quality ceiling.

Run:  python src/repair.py   (uses data/demo_disruption.json and results/plan.json)
"""
import itertools
from collections import defaultdict

from common import (HORIZON, STEP, duration, eligible_nodes, job_map, node_map, preds_of, release, succs_of,
                    to_hhmm, topo_order, violations)


# ------------------------------------------------------------------ residual capacity
class Timeline:
    """Tracks what is already booked on every node and licence, minute by minute (5-minute grid)."""

    def __init__(self, spec):
        self.spec = spec
        self.nodes = node_map(spec)
        self.jobs = job_map(spec)
        size = HORIZON + 600
        self.use = {n: {r: [0] * (size // STEP) for r in ("cores", "mem", "gpu")} for n in self.nodes}
        self.lic = {l: [0] * (size // STEP) for l in spec["licenses"]}
        self.blocked = {n: [False] * (size // STEP) for n in self.nodes}
        for n, node in self.nodes.items():
            for b0, b1, _ in node["blackouts"]:
                for k in range(max(0, b0) // STEP, min(size, b1) // STEP):
                    self.blocked[n][k] = True

    def fits(self, job, n, s, d=None):
        node = self.nodes[n]
        d = d if d is not None else duration(self.spec, job, node)
        if s < 0 or (s + d) // STEP >= len(self.blocked[n]):
            return False
        for k in range(s // STEP, (s + d) // STEP):
            if self.blocked[n][k]:
                return False
            for r in ("cores", "mem", "gpu"):
                if self.use[n][r][k] + job[r] > node[r]:
                    return False
            if job["license"] and self.lic[job["license"]][k] + 1 > self.spec["licenses"][job["license"]]["seats"]:
                return False
        return True

    def book(self, job, n, s, sign=1):
        d = duration(self.spec, job, self.nodes[n])
        for k in range(s // STEP, (s + d) // STEP):
            for r in ("cores", "mem", "gpu"):
                self.use[n][r][k] += sign * job[r]
            if job["license"]:
                self.lic[job["license"]][k] += sign

    def earliest(self, job, n, lb, limit=HORIZON + 300):
        s = ((lb + STEP - 1) // STEP) * STEP
        while s <= limit:
            if self.fits(job, n, s):
                return s
            s += STEP
        return None


# ------------------------------------------------------------------ which jobs can still move
def split_frozen(spec, plan):
    """Jobs that started before the disruption stay where they are, unless a new outage
    kills them mid-run (then they must restart)."""
    jobs, nodes = job_map(spec), node_map(spec)
    frozen, killed = {}, []
    for j, (n, s) in plan.items():
        if j not in jobs or s >= spec["as_of"]:
            continue
        e = s + duration(spec, jobs[j], nodes[n])
        hit = any(s < b1 and b0 < e and b0 >= spec["as_of"] for b0, b1, _ in nodes[n]["blackouts"])
        if hit:
            killed.append(j)
        else:
            frozen[j] = (n, s)
    return frozen, killed


# ------------------------------------------------------------------ 1. shift-right dispatcher
def dispatch(spec, fixed, order, preferred=None):
    """Place jobs in `order`, each at its earliest feasible time, on its preferred node if possible,
    otherwise on the node where it finishes first. Returns a full schedule."""
    jobs, nodes, preds = job_map(spec), node_map(spec), preds_of(spec)
    tl = Timeline(spec)
    sched = dict(fixed)
    for j, (n, s) in fixed.items():
        tl.book(jobs[j], n, s)
    for j in order:
        if j in sched:
            continue
        job = jobs[j]
        lb = max(release(spec, job), spec["as_of"])
        for p, lag in preds.get(j, []):
            if p in sched:
                lb = max(lb, sched[p][1] + duration(spec, jobs[p], nodes[sched[p][0]]) + lag)
        best = None
        pref = preferred.get(j) if preferred else None
        if pref and nodes.get(pref[0]) and pref[0] in [x["id"] for x in eligible_nodes(spec, job)]:
            s = tl.earliest(job, pref[0], max(lb, pref[1] if pref[1] >= spec["as_of"] else lb))
            if s is not None:
                best = (pref[0], s)
        if best is None:
            for node in eligible_nodes(spec, job):
                s = tl.earliest(job, node["id"], lb)
                if s is not None and (best is None or s + duration(spec, job, node) < best[1] + duration(spec, job, nodes[best[0]])):
                    best = (node["id"], s)
        if best is None:
            raise RuntimeError(f"Could not place {j}")
        sched[j] = best
        tl.book(job, *best)
    return sched


def shift_right(spec, plan):
    """Today's behaviour: keep the published plan's order and nodes, push jobs later as needed."""
    frozen, killed = split_frozen(spec, plan)
    jobs = job_map(spec)
    topo = {j: i for i, j in enumerate(topo_order(spec))}
    order = sorted((j for j in jobs if j not in frozen),
                   key=lambda j: (plan.get(j, ("", 10 ** 6))[1], topo[j]))
    order = [j for j in topo_order(spec) if j in set(order)]  # dependencies first, ties by plan time
    order.sort(key=lambda j: (max(plan.get(j, ("", 0))[1], 0), topo[j]))
    order = _respect_deps(spec, order)
    return dispatch(spec, frozen, order, preferred=plan), frozen, killed


def _respect_deps(spec, order):
    """Stable re-order so every job comes after its predecessors."""
    preds = preds_of(spec)
    placed, out, pending = set(), [], list(order)
    while pending:
        for i, j in enumerate(pending):
            if all(p in placed or p not in set(order) for p, _ in preds.get(j, [])):
                out.append(j)
                placed.add(j)
                pending.pop(i)
                break
    return out


# ------------------------------------------------------------------ critical-path helpers
def latest_finish(spec, movable):
    """Latest finish time (LFT) for each job so that no downstream deadline is missed, plus the
    weight of the deadline that binds. Classic critical-path analysis on the dependency graph."""
    jobs, nodes, succ = job_map(spec), node_map(spec), succs_of(spec)
    w = spec["weights"]
    min_dur = {j: min(duration(spec, jobs[j], n) for n in eligible_nodes(spec, jobs[j])) for j in jobs}
    lft, weight = {}, {}
    for j in reversed(topo_order(spec)):
        job = jobs[j]
        lft[j] = job["deadline"]
        weight[j] = w["priority"][job["priority"]] * (w["hard_multiplier"] if job["hard"] else 1.0)
        for s, lag in succ.get(j, []):
            if s in jobs and s in movable:
                cand = lft[s] - min_dur[s] - lag
                if cand < lft[j]:
                    lft[j] = cand
                weight[j] = max(weight[j], weight[s])
    return lft, weight


def descendants(spec, roots):
    succ = succs_of(spec)
    out, stack = set(), list(roots)
    while stack:
        j = stack.pop()
        for s, _ in succ.get(j, []):
            if s not in out:
                out.add(s)
                stack.append(s)
    return out


# ------------------------------------------------------------------ 2. hotspot sub-problem
SCARCE_POOLS = {"highmem", "gpu", "db"}


def find_hotspot(spec, schedule, frozen, max_jobs=6, seed_rank=0):
    """Jobs in the schedule that miss their latest-finish time and need a scarce resource, plus the
    other jobs competing for that same resource at overlapping times. Returns a list of job ids."""
    jobs, nodes = job_map(spec), node_map(spec)
    movable = set(jobs) - set(frozen)
    lft, weight = latest_finish(spec, movable)

    def scarce(j):
        return jobs[j]["license"] is not None or any(p in SCARCE_POOLS for p in jobs[j]["pools"])

    def span(j):
        n, s = schedule[j]
        return s, s + duration(spec, jobs[j], nodes[n])

    late = [j for j in sorted(movable) if scarce(j) and span(j)[1] > lft[j]]
    late.sort(key=lambda j: (-weight[j] * (span(j)[1] - lft[j]), j))  # ties broken by id: same answer every run
    if len(late) <= seed_rank:
        return []
    seed = late[seed_rank]  # 0 = the most damaging late job; 1, 2... = the next ones
    key = lambda j: ({p for p in jobs[j]["pools"] if p in SCARCE_POOLS}, jobs[j]["license"])
    pools, lic = key(seed)
    s0, e0 = span(seed)
    group = [seed]
    rivals = []
    for j in sorted(movable):
        if j == seed:
            continue
        jp, jl = key(j)
        on_scarce_node = nodes[schedule[j][0]]["pool"] in pools
        if on_scarce_node or (lic and jl == lic):
            s, e = span(j)
            gap = 0 if (s < e0 and s0 < e) else min(abs(s - e0), abs(s0 - e))
            rivals.append((gap, -weight[j], j))
    rivals.sort()
    for gap, _, j in rivals:
        if len(group) >= max_jobs or gap > 240:
            break
        group.append(j)
    return sorted(_closed(spec, group, frozen))


def _closed(spec, group, frozen):
    """Drop jobs that depend on the hotspot only THROUGH a job outside it: such a job's input time
    is not known until after the choice is made, so it cannot be given fixed options."""
    preds = preds_of(spec)
    group = set(group)
    changed = True
    while changed:
        changed = False
        floating = descendants(spec, group) - group - set(frozen)
        for j in sorted(group):
            if any(p in floating for p, _ in preds.get(j, [])):
                group.discard(j)
                changed = True
                break
    return group


def build_subproblem(spec, schedule, frozen, hotspot, options_per_job=3, plan=None):
    """Candidate (node, start) options for each hotspot job, with everything else held fixed.
    Jobs downstream of the hotspot are set aside and re-dispatched after the choice is made.
    Returns a dict describing the sub-problem (input to the exact solver and to the QUBO builder)."""
    jobs, nodes, preds = job_map(spec), node_map(spec), preds_of(spec)
    hs = set(hotspot)
    floating = (hs | descendants(spec, hs)) - set(frozen)
    fixed = {j: v for j, v in schedule.items() if j not in floating}
    tl = Timeline(spec)
    for j, (n, s) in fixed.items():
        tl.book(jobs[j], n, s)
    movable = set(jobs) - set(frozen)
    lft, weight = latest_finish(spec, movable)
    w = spec["weights"]

    def option_cost(j, n, s):
        job, node = jobs[j], nodes[n]
        d = duration(spec, job, node)
        c = weight[j] * max(0, s + d - lft[j])
        c += w["compute_cost"] * node["cost_per_hour"] * d / 60 + w["finish_early"] * (s + d)
        if plan and j in plan:
            c += w["move_node"] * (plan[j][0] != n) + w["move_minutes"] * abs(s - plan[j][1])
        return round(c, 3)

    order = [j for j in topo_order(spec) if j in hs]
    options = {}
    earliest_end = {}
    for j in order:
        job = jobs[j]
        lb = max(release(spec, job), spec["as_of"])
        for p, lag in preds.get(j, []):
            if p in fixed:
                lb = max(lb, fixed[p][1] + duration(spec, jobs[p], nodes[fixed[p][0]]) + lag)
            elif p in hs:
                lb = max(lb, earliest_end[p] + lag)
        cands = set()
        cands.add(tuple(schedule[j]))  # the current (shift-right) placement is always an option
        others_end = sorted({schedule[o][1] + duration(spec, jobs[o], nodes[schedule[o][0]]) for o in hs if o != j})
        for node in eligible_nodes(spec, job):
            s = tl.earliest(job, node["id"], lb)
            if s is not None:
                cands.add((node["id"], s))
            for t in others_end:  # start right after a rival finishes: a different ordering
                if t > lb:
                    s = tl.earliest(job, node["id"], t)
                    if s is not None:
                        cands.add((node["id"], s))
        ranked = sorted(cands, key=lambda o: (option_cost(j, *o), o))
        keep = [tuple(schedule[j])]
        per_node = defaultdict(int)
        per_node[schedule[j][0]] += 1
        for o in ranked:
            if len(keep) >= options_per_job:
                break
            if o not in keep and per_node[o[0]] < 2:
                keep.append(o)
                per_node[o[0]] += 1
        keep.sort(key=lambda o: (option_cost(j, *o), o))
        options[j] = [{"node": n, "start": s, "end": s + duration(spec, job, nodes[n]), "cost": option_cost(j, n, s)}
                      for n, s in keep]
        earliest_end[j] = min(o["end"] for o in options[j])

    # pairwise conflicts between options of different jobs
    edges = {(e["pred"], e["succ"]): e["lag"] for e in spec["edges"]}
    conflicts = []
    for a, b in itertools.combinations(order, 2):
        for ia, oa in enumerate(options[a]):
            for ib, ob in enumerate(options[b]):
                why = _pair_conflict(spec, tl, jobs, a, oa, b, ob, edges)
                if why:
                    conflicts.append({"a": [a, ia], "b": [b, ib], "why": why})
    return {"jobs": order, "options": options, "conflicts": conflicts, "fixed": fixed,
            "floating": sorted(floating - hs), "lft": {j: lft[j] for j in order},
            "weight": {j: weight[j] for j in order}}


def _pair_conflict(spec, tl, jobs, a, oa, b, ob, edges):
    if (a, b) in edges and oa["end"] + edges[(a, b)] > ob["start"]:
        return "dependency"
    if (b, a) in edges and ob["end"] + edges[(b, a)] > oa["start"]:
        return "dependency"
    lo, hi = max(oa["start"], ob["start"]), min(oa["end"], ob["end"])
    if lo >= hi:
        return None
    ja, jb = jobs[a], jobs[b]
    if oa["node"] == ob["node"]:
        node = tl.nodes[oa["node"]]
        for k in range(lo // STEP, hi // STEP):
            for r in ("cores", "mem", "gpu"):
                if tl.use[oa["node"]][r][k] + ja[r] + jb[r] > node[r]:
                    return f"{oa['node']} {r}"
    if ja["license"] and ja["license"] == jb["license"]:
        seats = spec["licenses"][ja["license"]]["seats"]
        for k in range(lo // STEP, hi // STEP):
            if tl.lic[ja["license"]][k] + 2 > seats:
                return ja["license"]
    return None


def choice_feasible(spec, sub, choice):
    """Exact check of a combination {job: option_index}, including 3-way capacity clashes
    that the pairwise conflict list cannot see. Problems that already exist among the fixed jobs alone
    (for example, jobs already running that overrun under worst-case runtimes) are not held against
    the combination: no choice can remove them."""
    jobs = job_map(spec)

    def check(trial):
        keep = set(trial)
        return set(violations({**spec, "held": spec["held"] + [j for j in jobs if j not in keep]}, trial))

    if "_baseline" not in sub:
        sub["_baseline"] = check(dict(sub["fixed"]))
    trial = dict(sub["fixed"])
    for j, i in choice.items():
        o = sub["options"][j][i]
        trial[j] = (o["node"], o["start"])
    return not (check(trial) - sub["_baseline"])


def choice_cost(sub, choice):
    return round(sum(sub["options"][j][i]["cost"] for j, i in choice.items()), 3)


def solve_exact(spec, sub):
    """Check every combination (fine up to ~10 jobs x 3 options). Returns the best feasible one."""
    best, best_cost, checked = None, None, 0
    ranges = [range(len(sub["options"][j])) for j in sub["jobs"]]
    for combo in itertools.product(*ranges):
        choice = dict(zip(sub["jobs"], combo))
        c = choice_cost(sub, choice)
        if best_cost is not None and c >= best_cost:
            continue
        checked += 1
        if choice_feasible(spec, sub, choice):
            best, best_cost = choice, c
    return {"choice": best, "cost": best_cost, "combinations": len(list(itertools.product(*ranges)))}


def complete(spec, sub, choice, plan=None):
    """Place the hotspot jobs as chosen, then re-dispatch everything downstream. Full schedule."""
    fixed = dict(sub["fixed"])
    for j, i in choice.items():
        o = sub["options"][j][i]
        fixed[j] = (o["node"], o["start"])
    order = [j for j in topo_order(spec) if j in set(sub["floating"])]
    return dispatch(spec, fixed, order, preferred=plan)


def describe(spec, sub):
    lines = [f"Hotspot: {len(sub['jobs'])} jobs, {sum(len(v) for v in sub['options'].values())} options, "
             f"{len(sub['conflicts'])} conflicting pairs; {len(sub['floating'])} downstream jobs re-dispatched after"]
    for j in sub["jobs"]:
        opts = ", ".join(f"{o['node']} {to_hhmm(o['start'])}-{to_hhmm(o['end'])} (cost {o['cost']:g})" for o in sub["options"][j])
        lines.append(f"  {j}: {opts}")
    return "\n".join(lines)


if __name__ == "__main__":
    from common import DATA, RESULTS, apply_disruption, load_json, load_reference, schedule_from_json, summary
    base = load_reference()
    plan = schedule_from_json(load_json(RESULTS / "plan.json"))
    spec = apply_disruption(base, load_json(DATA / "demo_disruption.json"))
    greedy, frozen, killed = shift_right(spec, plan)
    print("Shift-right repair:", summary(spec, greedy, plan))
    hs = find_hotspot(spec, greedy, frozen)
    sub = build_subproblem(spec, greedy, frozen, hs, options_per_job=4, plan=plan)
    print(describe(spec, sub))
    ex = solve_exact(spec, sub)
    print("Exact choice:", ex)
    full = complete(spec, sub, ex["choice"], plan)
    print("Hotspot repair (exact):", summary(spec, full, plan))
