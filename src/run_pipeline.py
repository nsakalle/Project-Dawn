"""The whole demo in one command:

  operator message -> watsonx.ai -> disruption JSON -> validate
  -> shift-right repair (today's behaviour) -> find the resource hotspot
  -> exact solve + QAOA solve of the hotspot -> re-dispatch downstream jobs
  -> full CP-SAT re-plan (quality ceiling) -> compare -> watsonx.ai briefing -> Gantt chart

Run (no IBM accounts):        python src/run_pipeline.py --mock
Run (watsonx + simulator):    python src/run_pipeline.py
Run (watsonx + IBM hardware): python src/run_pipeline.py --backend ibm
"""
import argparse
import datetime as dt

from common import (DATA, RESULTS, apply_disruption, load_json, load_reference, save_json, schedule_from_json,
                    schedule_table, schedule_to_json, summary, to_hhmm)
from explain import explain
from gantt import compare
from llm_to_spec import extract_disruption, validate_disruption
from planner import plan_night
from qubo_builder import tuned_qubo
from repair import build_subproblem, complete, describe, find_hotspot, shift_right, solve_exact


def banner(t):
    print(f"\n=== {t} " + "=" * max(0, 70 - len(t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request-file", default=str(DATA / "demo_request.txt"))
    ap.add_argument("--mock", action="store_true", help="skip watsonx.ai, use the saved demo answers")
    ap.add_argument("--backend", choices=["aer", "ibm"], default="aer")
    ap.add_argument("--ibm-backend", default=None)
    ap.add_argument("--options", type=int, default=4, help="candidate options per hotspot job")
    ap.add_argument("--max-hotspot", type=int, default=6)
    ap.add_argument("--skip-quantum", action="store_true")
    ap.add_argument("--as-of", default=None, help="current time HH:MM, if the message does not say")
    a = ap.parse_args()

    base = load_reference()
    plan_path = RESULTS / "plan.json"
    banner("0. Tonight's published plan (CP-SAT, made before the batch starts)")
    if plan_path.exists():
        plan = schedule_from_json(load_json(plan_path))
        print(f"Loaded {plan_path.name}")
    else:
        plan = plan_night(base)["schedule"]
        save_json(schedule_to_json(plan), plan_path)
    print(summary(base, plan))
    compare(base, [("Tonight's published plan: 56 jobs, every deadline met", plan)], RESULTS / "gantt-plan.png")

    banner("1. Operator message")
    text = open(a.request_file).read().strip()
    print(text)

    banner("2. watsonx.ai -> disruption JSON -> validation")
    d = extract_disruption(text, base, mock=a.mock)
    if a.as_of:
        d["as_of"] = a.as_of
    errs = validate_disruption(d, base)
    if not d.get("as_of"):
        errs.append("The message does not say what time it is now. Rerun with --as-of HH:MM.")
    if errs:
        print("Cannot repair yet. Ask the operator:")
        for e in errs:
            print(" -", e)
        return
    spec = apply_disruption(base, d)
    print(f"As of {to_hhmm(spec['as_of'])}. Assumptions: " + ("; ".join(spec["notes"]) or "none"))

    banner("3. Today's behaviour: shift the plan right")
    greedy, frozen, killed = shift_right(spec, plan)
    print(f"{len(frozen)} jobs already running or done; {len(killed)} killed by an outage and restarted")
    s_greedy = summary(spec, greedy, plan)
    print(s_greedy)

    banner("4. Hotspot: jobs fighting over a scarce resource")
    hs = find_hotspot(spec, greedy, frozen, max_jobs=a.max_hotspot)
    if not hs:
        print("No hotspot: shift-right already meets every critical-path deadline.")
        return
    sub = build_subproblem(spec, greedy, frozen, hs, a.options, plan)
    print(describe(spec, sub))

    banner("5a. Classical: exact solve of the hotspot")
    ex = solve_exact(spec, sub)
    print(f"Best combination costs {ex['cost']} (checked {ex['combinations']} combinations)")
    exact_full = complete(spec, sub, ex["choice"], plan)
    s_exact = summary(spec, exact_full, plan)
    print(s_exact)

    q = None
    final, method = exact_full, "hotspot solved exactly (classical)"
    if not a.skip_quantum:
        from quantum_solver import solve_qaoa
        banner(f"5b. Quantum: QAOA on the hotspot ({a.backend})")
        model, check = tuned_qubo(spec, sub, ex["cost"])
        print(f"QUBO: {model['qubo'].n} qubits, penalty {model['penalty']:g}; brute-force check {check}")
        q = solve_qaoa(spec, sub, model, ex["cost"], backend=a.backend, ibm_backend=a.ibm_backend)
        print(f"Best QAOA repair cost {q['cost']} (exact {ex['cost']}); valid samples {q['valid_share']:.1%} "
              f"vs random {q['random_valid_share']:.1%}; optimal samples {q['optimal_share']:.2%} "
              f"vs random {q['random_optimal_share']:.2%}")
        if q["choice"]:
            q_full = complete(spec, sub, q["choice"], plan)
            q["night"] = summary(spec, q_full, plan)
            print(q["night"])
            if q["night"]["score"] <= s_exact["score"] + 1e-6:
                final, method = q_full, f"hotspot solved by QAOA on {q.get('backend')}, checked against the exact answer"

    banner("6. Reference: full CP-SAT re-plan of every job not yet started")
    full = plan_night(spec, plan=plan, frozen=frozen, time_limit_s=20)
    s_full = summary(spec, full["schedule"], plan)
    print(s_full)

    banner("7. Comparison (lower score is better)")
    rows = [("Shift-right (today)", s_greedy), ("Hotspot, exact", s_exact)]
    if q and q.get("night"):
        rows.append(("Hotspot, QAOA", q["night"]))
    rows.append(("Full CP-SAT re-plan", s_full))
    print(f"{'Approach':<22}{'Score':>9}{'SLA misses':>12}{'Hard':>6}{'Min late':>10}{'Jobs moved':>12}{'Ends':>7}")
    for name, s in rows:
        print(f"{name:<22}{s['score']:>9}{s['sla_misses']:>12}{s['hard_misses']:>6}{s['minutes_late']:>10}{s['jobs_moved']:>12}{s['batch_finishes']:>7}")

    banner("8. Briefing for the operator (watsonx.ai)")
    brief = explain(spec, d, plan, greedy, final, method, hs, mock=a.mock)
    print(brief)
    print("\nChanged jobs:")
    print(schedule_table(spec, final, greedy, only=[j for j in final if tuple(final[j]) != tuple(greedy.get(j, ()))]))

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    png = compare(spec, [(f"Today: shift-right repair ({s_greedy['sla_misses']} deadline misses, {s_greedy['hard_misses']} hard)", greedy),
                         (f"Hybrid repair: {method} ({s_exact['sla_misses'] if final is exact_full else q['night']['sla_misses']} misses)", final)],
                  RESULTS / f"gantt-{stamp}.png", focus=set(hs))
    save_json({"message": text, "disruption": d, "hotspot": hs, "subproblem": sub,
               "summaries": {k: v for k, v in rows}, "quantum": q, "briefing": brief,
               "shift_right": schedule_to_json(greedy), "final": schedule_to_json(final)},
              RESULTS / f"run-{stamp}.json")
    print(f"\nSaved results/run-{stamp}.json and {png.name}")


if __name__ == "__main__":
    main()
