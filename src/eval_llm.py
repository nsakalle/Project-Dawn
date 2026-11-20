"""Measure how accurately watsonx.ai turns operator messages into the right disruption JSON.

Test cases live in data/llm_test_set.jsonl, one per line: {"message": "...", "expected": {...}}.
Add your own cases to grow it, including vague wording, typos and names that are not in the data.

Run:  python src/eval_llm.py               (needs watsonx.ai configured)
      python src/eval_llm.py --offline     (scores the offline reader, no watsonx needed)
      python src/eval_llm.py --self-check  (no watsonx; feeds the expected answers back to test the scorer)
"""
import argparse
import json

from common import DATA, load_reference
from llm_to_spec import extract_disruption, validate_disruption
import watsonx_client

FIELDS = ["as_of", "external_delays", "node_outages", "deadline_changes", "priority_changes", "job_holds",
          "use_p90_runtimes"]
IGNORE = {"reason"}  # free-text details that do not change the schedule


def norm(field, value):
    """Comparable form, ignoring list order and free-text extras such as an outage's reason."""
    if field == "as_of":
        return value or None
    if field == "use_p90_runtimes":
        return bool(value)
    if field == "job_holds":
        return frozenset(value or [])
    return frozenset(json.dumps({k: v for k, v in sorted(d.items()) if k not in IGNORE}) for d in (value or []))


def compare(field, got, exp):
    if field == "deadline_changes":
        # deadline_type only counts when the test case states it
        g = {c["job_id"]: c for c in (got or [])}
        e = {c["job_id"]: c for c in (exp or [])}
        if set(g) != set(e):
            return False
        return all(g[k].get("deadline") == e[k]["deadline"] and
                   ("deadline_type" not in e[k] or g[k].get("deadline_type") == e[k]["deadline_type"]) for k in e)
    return norm(field, got) == norm(field, exp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--offline", action="store_true", help="score the offline reader instead of watsonx.ai")
    a = ap.parse_args()
    if not a.self_check and not a.offline and not watsonx_client.configured():
        raise SystemExit("watsonx.ai is not configured. Fill in .env, or run with --self-check.")
    spec = load_reference()
    cases = [json.loads(l) for l in open(DATA / "llm_test_set.jsonl") if l.strip()]
    exact, hits, invalid = 0, {f: 0 for f in FIELDS}, 0
    for i, c in enumerate(cases, 1):
        mode = "offline" if a.offline else "watsonx"
        got = c["expected"] if a.self_check else extract_disruption(c["message"], spec, mode=mode)
        if validate_disruption(got, spec):
            invalid += 1
        wrong = [f for f in FIELDS if not compare(f, got.get(f), c["expected"].get(f))]
        for f in FIELDS:
            hits[f] += f not in wrong
        exact += not wrong
        print(f"{i:>2}. {'OK  ' if not wrong else 'MISS'} {c['message'][:64]}" + ("" if not wrong else f"  -> wrong: {', '.join(wrong)}"))
    n = len(cases)
    print(f"\nExactly right: {exact}/{n} ({exact / n:.0%}); failed validation: {invalid}")
    for f in FIELDS:
        print(f"  {f:<18} {hits[f]}/{n}")


if __name__ == "__main__":
    main()
