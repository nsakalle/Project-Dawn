"""Final step: a short plain-English briefing for the operator, written by watsonx.ai from verified
facts only. Without watsonx configured, a template briefing is produced instead."""
import json

from common import duration, job_map, node_map, sla_report, to_hhmm
import watsonx_client

SYSTEM = """You brief a night-shift batch operations lead on a repaired schedule.
Use ONLY the facts in the JSON you are given. Do not invent jobs, times, numbers or reasons.
Write 4 to 7 short sentences of plain English, no bullet points, no JSON:
what changed tonight, which deadlines the old approach would have missed, what the repair moves
(job, from, to), whether every deadline is now met, and every assumption that was made."""


def facts(spec, disruption, plan, before, after, method, hotspot):
    jobs, nodes = job_map(spec), node_map(spec)

    def misses(s):
        return [{"job": j, "name": jobs[j]["name"], "minutes_late": m, "hard_deadline": h} for j, m, h in sla_report(spec, s)]

    moves = []
    for j in sorted(after):
        if j in before and tuple(before[j]) != tuple(after[j]):
            (bn, bs), (an, as_) = before[j], after[j]
            moves.append({"job": j, "name": jobs[j]["name"],
                          "from": f"{bn} at {to_hhmm(bs)}", "to": f"{an} at {to_hhmm(as_)}",
                          "now_ends": to_hhmm(as_ + duration(spec, jobs[j], nodes[an])),
                          "deadline": to_hhmm(jobs[j]["deadline"])})
    return {"as_of": to_hhmm(spec["as_of"]), "disruption": disruption,
            "held_jobs": spec["held"], "notes": spec["notes"],
            "misses_with_shift_right": misses(before), "misses_after_repair": misses(after),
            "repair_changes_vs_shift_right": moves, "repair_method": method, "hotspot_jobs": hotspot}


def _count(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def template(f):
    d = f["disruption"]
    parts = [_count(len(d.get("external_delays", [])), "late feed"), _count(len(d.get("node_outages", [])), "server outage"),
             _count(len(d.get("deadline_changes", [])), "deadline change")]
    parts = [p for p in parts if not p.startswith("0 ")]
    out = [f"As of {f['as_of']}: " + (", ".join(parts) or "no feed, server or deadline changes") +
           (f"; skipped tonight: {', '.join(f['held_jobs'])}." if f["held_jobs"] else ".")]
    if f["misses_with_shift_right"]:
        out.append("Simply shifting the plan later would miss " + "; ".join(
            f"{m['job']} ({m['name']}) by {m['minutes_late']} min{', a hard deadline' if m['hard_deadline'] else ''}"
            for m in f["misses_with_shift_right"]) + ".")
    for m in f["repair_changes_vs_shift_right"]:
        same_node = m["from"].split(" at ")[0] == m["to"].split(" at ")[0]
        if same_node:
            out.append(f"{m['job']} ({m['name']}) now starts at {m['to'].split(' at ')[1]} on {m['to'].split(' at ')[0]}, "
                       f"ending {m['now_ends']} against a {m['deadline']} deadline.")
        else:
            out.append(f"{m['job']} ({m['name']}) moves from {m['from']} to {m['to']}, "
                       f"ending {m['now_ends']} against a {m['deadline']} deadline.")
    out.append("Every deadline is now met." if not f["misses_after_repair"] else
               "Still late: " + "; ".join(f"{m['job']} by {m['minutes_late']} min" for m in f["misses_after_repair"]) + ".")
    out.append(f"Repair method: {f['repair_method']}.")
    if f["notes"]:
        out.append("Assumptions: " + "; ".join(f["notes"]) + ".")
    return " ".join(out)


def explain(spec, disruption, plan, before, after, method, hotspot, mock=False):
    f = facts(spec, disruption, plan, before, after, method, hotspot)
    if mock or not watsonx_client.configured():
        return template(f)
    return watsonx_client.chat(SYSTEM, json.dumps(f, indent=2), max_tokens=500).strip()
