"""Step 1: turn an operator's plain-English message about tonight into the disruption JSON,
using watsonx.ai (Granite).

The model only extracts WHAT CHANGED (late feeds, node outages, new deadlines, holds...). The fixed
facts come from the CSV files and are merged in common.apply_disruption().

Run (live):  python src/llm_to_spec.py data/demo_request.txt
Run (mock):  python src/llm_to_spec.py data/demo_request.txt --mock     (no watsonx needed)
"""
import argparse
import json
import re

from common import DATA, apply_disruption, load_json, load_reference, to_hhmm
import watsonx_client

SYSTEM_PROMPT = """You turn a batch operations message into JSON describing tonight's disruptions.
Return ONLY one JSON object, no prose, with exactly these keys:
{
  "as_of": "HH:MM time the message was sent, or null if not stated",
  "external_delays": [{"ext_id": "...", "new_time": "HH:MM"}],
  "node_outages": [{"node_id": "...", "start": "HH:MM", "end": "HH:MM", "reason": "..."}],
  "deadline_changes": [{"job_id": "...", "deadline": "HH:MM", "deadline_type": "hard" or "soft"}],
  "priority_changes": [{"job_id": "...", "priority": "P1" | "P2" | "P3" | "P4"}],
  "job_holds": ["job_id", ...],
  "use_p90_runtimes": true or false,
  "assumptions": ["each interpretation you made of vague wording, in plain English"]
}
Rules:
- Use only IDs that appear in the DATA section. Never invent IDs. Match descriptions to IDs
  (for example "closing prices feed" -> the external event whose description mentions closing prices).
- All times are 24-hour HH:MM. The batch window runs 20:00 to 08:00, so "1am" is "01:00".
- If a delay is given as a duration ("90 minutes late"), add it to the event's expected time.
  If both a duration and a time are given, use the time.
- If a new deadline does not say hard or soft, keep the job's current deadline type.
- "Hold", "skip", "don't run" a job -> job_holds.
- "Plan for worst-case runtimes" or similar -> use_p90_runtimes true. Otherwise false.
- Put every interpretation of vague wording in "assumptions".
- Lists with nothing to report are empty lists []."""


def reference_as_text(spec):
    L = ["NODES (id, name, pool):"]
    L += [f"{n['id']}, {n['name']}, {n['pool']}" for n in spec["nodes"]]
    L.append("EXTERNAL EVENTS (id, description, source, expected time):")
    L += [f"{k}, {e['description']}, {e['source']}, {to_hhmm(e['expected'])}" for k, e in spec["external"].items()]
    L.append("JOBS (id, name, application, priority, deadline, deadline type):")
    L += [f"{j['id']}, {j['name']}, {j['app']}, {j['priority']}, {to_hhmm(j['deadline'])}, "
          f"{'hard' if j['hard'] else 'soft'}" for j in spec["jobs"]]
    return "\n".join(L)


def extract_disruption(text, spec, mock=False, mode=None, default_as_of=None):
    """mode: 'watsonx' (Granite), 'offline' (rule-based reader, no AI), or 'demo' (the saved demo answer).
    Without a mode: watsonx.ai if it is configured, otherwise the offline reader."""
    if mock:
        mode = "demo"
    if mode is None:
        mode = "watsonx" if watsonx_client.configured() else "offline"
    if mode == "demo":
        return load_json(DATA / "demo_disruption.json")
    if mode == "offline":
        from rule_parser import parse_message
        d = parse_message(text, spec, default_as_of)
        d.pop("_as_of_from_message", None)
        return d
    reply = watsonx_client.chat(SYSTEM_PROMPT, f"DATA\n{reference_as_text(spec)}\n\nMESSAGE\n{text}", max_tokens=900)
    return watsonx_client.extract_json(reply)


TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def validate_disruption(d, spec):
    """Catch model mistakes before they reach the solvers. Empty list = OK."""
    errs = []
    jobs = {j["id"] for j in spec["jobs"]}
    nodes = {n["id"] for n in spec["nodes"]}

    def t(v, what):
        if not isinstance(v, str) or not TIME.match(v):
            errs.append(f"{what}: '{v}' is not an HH:MM time.")
            return False
        return True

    if d.get("as_of"):
        t(d["as_of"], "as_of")
    for e in d.get("external_delays", []):
        if e.get("ext_id") not in spec["external"]:
            errs.append(f"Unknown external event {e.get('ext_id')}.")
        t(e.get("new_time"), f"new time for {e.get('ext_id')}")
    for o in d.get("node_outages", []):
        if o.get("node_id") not in nodes:
            errs.append(f"Unknown node {o.get('node_id')}.")
        if t(o.get("start"), "outage start") and t(o.get("end"), "outage end"):
            from common import to_min
            if to_min(o["end"]) <= to_min(o["start"]):
                errs.append(f"Outage on {o.get('node_id')} ends before it starts.")
    for c in d.get("deadline_changes", []):
        if c.get("job_id") not in jobs:
            errs.append(f"Unknown job {c.get('job_id')} in deadline change.")
        t(c.get("deadline"), f"deadline for {c.get('job_id')}")
    for c in d.get("priority_changes", []):
        if c.get("job_id") not in jobs:
            errs.append(f"Unknown job {c.get('job_id')} in priority change.")
        if c.get("priority") not in ("P1", "P2", "P3", "P4"):
            errs.append(f"Unknown priority {c.get('priority')}.")
    for h in d.get("job_holds", []):
        if h not in jobs:
            errs.append(f"Unknown job {h} in holds.")
    return errs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("request_file", nargs="?", default=str(DATA / "demo_request.txt"))
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    base = load_reference()
    text = open(a.request_file).read().strip()
    d = extract_disruption(text, base, mock=a.mock)
    print("Disruption JSON:\n" + json.dumps(d, indent=2))
    errs = validate_disruption(d, base)
    if errs:
        print("\nProblems to clarify with the operator:")
        for e in errs:
            print(" -", e)
    else:
        spec = apply_disruption(base, d)
        print(f"\nValid. As of {to_hhmm(spec['as_of'])}; held jobs: {spec['held'] or 'none'}")
