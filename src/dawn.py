"""Project Dawn engine: one function that runs the whole repair for one operator message.
Used by the app (app.py) and by the command line (run_pipeline.py).

    from dawn import Settings, run_dawn
    result = run_dawn("It's 23:15. HMEM-01 is down from 01:00 to 04:00.", Settings())
"""
import time
from dataclasses import dataclass

from common import (RESULTS, apply_disruption, duration, job_map, load_json, load_reference, node_map,
                    save_json, schedule_from_json, schedule_to_json, sla_report, summary, to_hhmm)
from explain import explain
from llm_to_spec import extract_disruption, validate_disruption
from planner import plan_night
from repair import build_subproblem, choice_cost, complete, find_hotspot, shift_right, solve_exact

MAX_TRAIN_QUBITS = 16  # keeps QAOA training to about 30 seconds on a laptop, for a live demo


@dataclass
class Settings:
    reader: str = "auto"          # "watsonx", "offline" (no AI) or "auto" (watsonx if configured)
    backend: str = "aer"          # "aer" (local simulator) or "ibm" (real IBM quantum computer)
    ibm_token: str = ""
    ibm_instance: str = ""
    ibm_backend: str = ""         # empty = least busy machine
    default_as_of: str = "23:15"  # used when the message does not say what time it is
    options_per_job: int = 4
    max_hotspot: int = 6
    reps: int = 3
    shots: int = 4096
    run_quantum: bool = True
    full_replan: bool = True


def _noop(stage, message):
    pass


def _plan(base):
    path = RESULTS / "plan.json"
    if path.exists():
        return schedule_from_json(load_json(path))
    plan = plan_night(base)["schedule"]
    save_json(schedule_to_json(plan), path)
    return plan


def _placement(spec, sched, j):
    jobs, nodes = job_map(spec), node_map(spec)
    if j not in sched:
        return None
    n, s = sched[j]
    e = s + duration(spec, jobs[j], nodes[n])
    late = e - jobs[j]["deadline"]
    return {"node": n, "start": to_hhmm(s), "end": to_hhmm(e), "late_min": max(0, late),
            "status": "on time" if late <= 0 else f"{late} min late" + (" (hard)" if jobs[j]["hard"] else "")}


def run_dawn(text, settings=None, progress=_noop):
    s = settings or Settings()
    t_start = time.perf_counter()
    out = {"message": text, "settings": {k: v for k, v in s.__dict__.items() if "token" not in k},
           "errors": [], "notes": [], "timings": {}}
    base = load_reference()
    plan = _plan(base)

    # 1. read the message
    progress("read", "Reading the message" + (" with watsonx.ai" if s.reader == "watsonx" else ""))
    t = time.perf_counter()
    mode = None if s.reader == "auto" else s.reader
    try:
        d = extract_disruption(text, base, mode=mode, default_as_of=s.default_as_of)
    except Exception as e:  # watsonx.ai unreachable, bad key, model reply without JSON...
        out["notes"].append(f"watsonx.ai failed ({str(e)[:160]}); used the offline reader instead")
        d = extract_disruption(text, base, mode="offline", default_as_of=s.default_as_of)
        mode = "offline"
    out["reader_used"] = mode or ("watsonx" if __import__("watsonx_client").configured() else "offline")
    if not d.get("as_of") and s.default_as_of:
        d["as_of"] = s.default_as_of
        d.setdefault("assumptions", []).append(f"time not given in the message; used {s.default_as_of}")
    out["disruption"] = d
    out["timings"]["read"] = round(time.perf_counter() - t, 2)
    errs = validate_disruption(d, base)
    if not any(d.get(k) for k in ("external_delays", "node_outages", "deadline_changes", "priority_changes", "job_holds")) \
            and not d.get("use_p90_runtimes"):
        errs.append("No changes were found in the message. Try naming the feed, server or job, and a time.")
    if errs:
        out["errors"] = errs
        return out
    spec = apply_disruption(base, d)
    out["as_of"] = to_hhmm(spec["as_of"])
    out["held"] = spec["held"]

    # 2. today's behaviour
    progress("shift", "Shifting the plan right, as today's scheduler would")
    t = time.perf_counter()
    greedy, frozen, killed = shift_right(spec, plan)
    out["timings"]["shift_right"] = round(time.perf_counter() - t, 2)
    out["frozen"], out["killed"] = len(frozen), killed
    schedules = {"plan": plan, "shift_right": greedy}
    summaries = {"Published plan": summary(base, plan), "Shift-right (today)": summary(spec, greedy, plan)}

    # 3. hotspot
    progress("hotspot", "Finding the jobs fighting over scarce resources")
    opts, size = s.options_per_job, s.max_hotspot
    hs = find_hotspot(spec, greedy, frozen, max_jobs=size)
    sub = None
    while hs:
        sub = build_subproblem(spec, greedy, frozen, hs, opts, plan)
        n = sum(len(v) for v in sub["options"].values())
        if not s.run_quantum or n <= MAX_TRAIN_QUBITS:
            break
        if opts > 2:
            opts -= 1
        elif size > 2:
            size -= 1
            hs = find_hotspot(spec, greedy, frozen, max_jobs=size)
        else:
            break
        out["notes"].append(f"Hotspot trimmed to fit {MAX_TRAIN_QUBITS} qubits, so QAOA trains in about 30 seconds "
                            f"({len(hs)} jobs, {opts} options each)")

    trims = [n for n in out["notes"] if n.startswith("Hotspot trimmed")]
    out["notes"] = [n for n in out["notes"] if not n.startswith("Hotspot trimmed")] + trims[-1:]
    final, method = greedy, "shift-right only"
    if not hs:
        out["notes"].append("No hotspot: no late job is waiting for a scarce resource, so there is nothing for the "
                            "quantum solver to decide. The full re-plan below shows whether anything can be improved.")
    else:
        out["hotspot"] = {"jobs": sub["jobs"], "qubits": sum(len(v) for v in sub["options"].values()),
                          "conflicts": len(sub["conflicts"]),
                          "options": {j: [{**o, "start": to_hhmm(o["start"]), "end": to_hhmm(o["end"])} for o in v]
                                      for j, v in sub["options"].items()},
                          "combinations": 1}
        for v in sub["options"].values():
            out["hotspot"]["combinations"] *= len(v)

        # 4a. classical exact
        progress("classical", f"Classical: checking all {out['hotspot']['combinations']} combinations")
        t = time.perf_counter()
        ex = solve_exact(spec, sub)
        greedy_choice = {j: next(i for i, o in enumerate(sub["options"][j]) if (o["node"], o["start"]) == tuple(greedy[j]))
                         for j in sub["jobs"]}
        if ex["choice"] is None:  # should not happen; keep today's placement rather than fail
            out["notes"].append("The exact solver found no valid combination; kept today's placement for the hotspot")
            ex = {"choice": greedy_choice, "cost": choice_cost(sub, greedy_choice), "combinations": ex["combinations"]}
        exact_full = complete(spec, sub, ex["choice"], plan)
        out["timings"]["classical"] = round(time.perf_counter() - t, 2)
        out["classical"] = {"cost": ex["cost"], "shift_right_cost": choice_cost(sub, greedy_choice),
                            "seconds": out["timings"]["classical"], "choice": ex["choice"]}
        schedules["classical"] = exact_full
        summaries["Classical repair"] = summary(spec, exact_full, plan)
        final, method = exact_full, "classical exact solver"

        # 4b. quantum
        if s.run_quantum:
            from qubo_builder import tuned_qubo
            from quantum_solver import solve_qaoa

            where = "IBM quantum computer" if s.backend == "ibm" else "local quantum simulator"
            progress("quantum", f"Quantum: training QAOA ({out['hotspot']['qubits']} qubits) on the simulator")
            t = time.perf_counter()
            model, check = tuned_qubo(spec, sub, ex["cost"])

            def submitted(job_id, backend_name):
                out["ibm_job"] = {"job_id": job_id, "backend": backend_name}
                progress("quantum", f"Quantum: job {job_id} sent to {backend_name}; waiting in IBM's queue")

            try:
                if s.backend == "ibm":  # check the key before spending a minute training
                    progress("quantum", "Quantum: checking the IBM Quantum API key")
                    from ibm_quantum import get_service
                    get_service(s.ibm_token, s.ibm_instance)
                    progress("quantum", f"Quantum: training QAOA ({out['hotspot']['qubits']} qubits) on the simulator")
                q = solve_qaoa(spec, sub, model, ex["cost"], s.reps, s.shots, backend=s.backend,
                               ibm_backend=s.ibm_backend or None, ibm_token=s.ibm_token, ibm_instance=s.ibm_instance,
                               on_submit=submitted)
                q["qubo_check"] = check
                q["where"] = where
                q["seconds"] = round(time.perf_counter() - t, 2)
                if q["choice"]:
                    q_full = complete(spec, sub, q["choice"], plan)
                    schedules["quantum"] = q_full
                    summaries["Quantum repair"] = summary(spec, q_full, plan)
                    q["matches_classical"] = q["cost"] <= ex["cost"] + 1e-6
                    if q["matches_classical"]:
                        final, method = q_full, f"QAOA on the {where}, checked against the classical answer"
                q.pop("angles", None)
                out["quantum"] = q
            except Exception as e:
                out["quantum"] = {"error": str(e)[:400], "where": where}
            out["timings"]["quantum"] = round(time.perf_counter() - t, 2)

        # decision table: where each hotspot job runs under each approach
        rows = []
        jobs = job_map(spec)
        for j in sub["jobs"]:
            rows.append({"job": j, "name": jobs[j]["name"], "deadline": to_hhmm(jobs[j]["deadline"]),
                         "hard": jobs[j]["hard"], "today": _placement(spec, greedy, j),
                         "classical": _placement(spec, exact_full, j),
                         "quantum": _placement(spec, schedules["quantum"], j) if "quantum" in schedules else None})
        out["decisions"] = rows

    # 5. full re-plan (quality ceiling)
    if s.full_replan:
        progress("replan", "Classical full re-plan of every job not yet started (reference)")
        t = time.perf_counter()
        full = plan_night(spec, plan=plan, frozen=frozen, time_limit_s=15)
        out["timings"]["full_replan"] = round(time.perf_counter() - t, 2)
        if full["schedule"]:
            schedules["full_replan"] = full["schedule"]
            summaries["Full CP-SAT re-plan"] = summary(spec, full["schedule"], plan)

    # 6. briefing + charts
    progress("brief", "Writing the briefing")
    t = time.perf_counter()
    try:
        out["briefing"] = explain(spec, d, plan, greedy, final, method, hs, mock=(out["reader_used"] != "watsonx"))
    except Exception as e:
        out["notes"].append(f"watsonx.ai briefing failed ({str(e)[:120]}); used the template briefing")
        out["briefing"] = explain(spec, d, plan, greedy, final, method, hs, mock=True)
    out["timings"]["briefing"] = round(time.perf_counter() - t, 2)
    out["method"] = method
    out["summaries"] = summaries
    out["misses"] = {k: [{"job": j, "minutes_late": m, "hard": h} for j, m, h in sla_report(spec, v)]
                     for k, v in schedules.items() if k != "plan"}
    out["schedules"] = {k: schedule_to_json(v) for k, v in schedules.items()}
    out["_spec"] = spec  # for drawing charts; not saved
    out["timings"]["total"] = round(time.perf_counter() - t_start, 2)
    progress("done", "Done")
    return out


def draw_charts(result, folder=None, stamp=None):
    """Gantt PNGs for the app: one per approach. Returns {name: path}."""
    from gantt import compare
    folder = folder or RESULTS
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    spec = result["_spec"]
    from common import load_reference
    paths = {}
    base = load_reference()
    paths["plan"] = str(compare(base, [("Tonight's published plan, before anything went wrong",
                                        schedule_from_json(result["schedules"]["plan"]))],
                                folder / f"gantt-plan-{stamp}.png"))
    titles = {"shift_right": "Today: shift-right repair", "classical": "Classical repair (exact solver)",
              "quantum": "Quantum repair (QAOA)", "full_replan": "Full CP-SAT re-plan (reference)"}
    focus = set(result.get("hotspot", {}).get("jobs", []))
    for key, title in titles.items():
        if key in result["schedules"]:
            sched = schedule_from_json(result["schedules"][key])
            m = result["misses"].get(key, [])
            label = f"{title}: " + (f"{len(m)} deadline misses ({sum(1 for x in m if x['hard'])} hard)" if m else "every deadline met")
            paths[key] = str(compare(spec, [(label, sched)], folder / f"gantt-{key}-{stamp}.png", focus=focus))
    return paths


def save_result(result, path):
    save_json({k: v for k, v in result.items() if not k.startswith("_")}, path)
