"""Offline message reader: turns an operator message into the disruption JSON using simple rules,
with no AI model. Used when watsonx.ai is not connected, so the demo works with any wording.

It understands the common patterns: a time stamp ("It's 23:15"), late feeds ("FX file 45 minutes
late", "POS files until 02:30"), server outages ("HMEM-01 down 01:00 to 04:00"), new deadlines
("VaR by 04:30"), holds ("skip the fraud retrain"), priority changes ("bump churn scoring to P2")
and cautious runtimes ("plan for worst-case runtimes"). Anything it reads loosely is listed in
"assumptions", exactly like the watsonx.ai answer.

watsonx.ai is far better with unusual wording; this is the safety net, not the main act.
"""
import re

from common import to_hhmm, to_min

NODE_RE = re.compile(r"\b(GEN|HMEM|GPU|DB|XFER)[\s-]?0?(\d)\b", re.I)
TIME_RE = re.compile(r"\b(?:(\d{1,2}):(\d{2})\s*(am|pm)?|(\d{1,2})\s*(am|pm)|(midnight|noon))\b", re.I)
DUR_RE = re.compile(r"\b(\d+(?:\.\d+)?|an?|one|two|three|four|half an?)\s*(hours?|hrs?|h|minutes?|mins?)\b", re.I)
WORDNUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4}

EXT_ALIASES = {
    "EXT-MKT": ["closing price", "market data", "positions feed", "prices feed", "price feed", "positions file"],
    "EXT-FX": ["fx", "exchange rate"],
    "EXT-POS-US": ["us pos", "us store", "us sales", "us point-of-sale", "us point of sale"],
    "EXT-POS-EU": ["eu pos", "eu store", "eu sales", "eu point-of-sale", "eu point of sale"],
    "EXT-ERP-CLOSE": ["erp", "ledger close"],
    "EXT-CRM-API": ["crm export", "crm api", "crm bulk", "crm window"],
    "EXT-HR-TS": ["timesheet"],
    "EXT-DB-SNAP": ["snapshot"],
    "EXT-WEATHER": ["weather"],
    "EXT-SUPP-EDI": ["supplier edi", "edi message", "supplier inventory", "supplier"],
    "EXT-WEBLOGS": ["log rotation", "web log", "weblog"],
    "EXT-BANK-WIN": ["sftp window", "bank window", "bank sftp", "intake window"],
}
JOB_ALIASES = {
    "RSK03": ["var", "value-at-risk", "value at risk"],
    "RSK04": ["stress test"],
    "RSK06": ["regulatory report", "regulatory risk report", "risk report"],
    "PAY04": ["bank file", "payroll file"],
    "PAY02": ["payroll calc", "gross-to-net", "gross to net"],
    "FIN08": ["management pack"],
    "FIN07": ["consolidation"],
    "SCM03": ["replenishment"],
    "SCM04": ["purchase order"],
    "ML07": ["fraud"],
    "OPS07": ["log analytics", "log indexing"],
    "OPS08": ["temp cleanup", "temp space", "cleanup"],
    "SLS07": ["sales dashboard"],
    "SLS08": ["store dashboard", "store performance"],
}
DELAY_WORDS = r"(late|delay|behind|won'?t|not until|until|arriv|expect|opens?|slip|down|held up|running|early|ahead|land)"
OUTAGE_WORDS = r"(down|offline|outage|patch|maintenance|unavailable|fail|reboot|out of service|take it out)"
HOLD_WORDS = r"\b(skip|hold|don'?t run|do not run|cancel|pause|drop|no need (?:for|to run))\b"
STOP = {"the", "and", "of", "from", "to", "a", "for", "on", "in", "db", "job", "run", "data", "file", "files"}


def _times(sentence):
    """All clock times in a sentence, as (minute-of-window, position, text). 'not 02:00' is skipped."""
    out = []
    for m in TIME_RE.finditer(sentence):
        before = sentence[max(0, m.start() - 12):m.start()].lower()
        if re.search(r"(not|instead of|than|was|from the)\s*$", before):
            continue
        if m.group(1):
            h, mi, ap = int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower()
        elif m.group(4):
            h, mi, ap = int(m.group(4)), 0, m.group(5).lower()
        else:
            h, mi, ap = (0, 0, "") if m.group(6).lower() == "midnight" else (12, 0, "")
        if ap == "pm" and h < 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        if h > 23 or mi > 59:
            continue
        out.append((f"{h:02d}:{mi:02d}", m.start(), m.group(0)))
    return out


def _duration(sentence):
    m = DUR_RE.search(sentence)
    if not m:
        return None, None
    num, unit = m.group(1).lower(), m.group(2).lower()
    if num.startswith("half"):
        val = 0.5
    else:
        val = WORDNUM.get(num, None)
        val = float(num) if val is None else val
    minutes = val * 60 if unit.startswith("h") else val
    return int(round(minutes)), m.group(0)


def _sentences(text):
    parts = re.split(r"(?<=[.!?;])\s+|\n+|\s+-\s+(?=[A-Za-z])", text)
    return [p.strip() for p in parts if p and p.strip()]


def _find_jobs(spec, sentence):
    s = sentence.lower()
    found = {}
    for j in spec["jobs"]:
        if re.search(rf"\b{j['id'].lower()}\b", s):
            found[j["id"]] = 1.0
    for jid, aliases in JOB_ALIASES.items():
        for a in aliases:
            if re.search(rf"\b{re.escape(a)}", s):
                found[jid] = max(found.get(jid, 0), 0.95)
    for j in spec["jobs"]:
        words = [w for w in re.findall(r"[a-z0-9-]+", j["name"].lower()) if w not in STOP]
        if not words:
            continue
        hit = sum(1 for w in words if re.search(rf"\b{re.escape(w)}", s))
        score = hit / len(words)
        if hit >= 2 and score >= 0.6 or score == 1.0 and len(words) >= 2:
            found[j["id"]] = max(found.get(j["id"], 0), score * 0.9)
    if not found:
        return []
    best = max(found.values())
    # keep strong matches only; two different jobs can both be named in one sentence
    return sorted(j for j, v in found.items() if v >= max(0.6, best - 0.2))


def _find_ext(sentence):
    s = sentence.lower()
    hits = []
    for eid, aliases in EXT_ALIASES.items():
        for a in aliases:
            if re.search(rf"\b{re.escape(a)}", s):
                hits.append((len(a), eid))
                break
    if not hits:
        return None
    return max(hits)[1]  # longest alias wins (e.g. "supplier edi" over "edi")


def parse_message(text, spec, default_as_of=None):
    d = {"as_of": None, "external_delays": [], "node_outages": [], "deadline_changes": [], "priority_changes": [],
         "job_holds": [], "use_p90_runtimes": False, "assumptions": []}
    notes = d["assumptions"]
    nodes = {n["id"] for n in spec["nodes"]}

    # 1. the time stamp: "It's 23:15", "23:15 update:", "00:30 - ..."
    m = re.search(r"\b(?:it'?s|it is|as of|time is)\s+(\d{1,2}:\d{2}\s*(?:am|pm)?|midnight)", text, re.I) or \
        re.match(r"\s*(\d{1,2}:\d{2})\b", text)
    if m:
        t = _times(m.group(1))
        if t:
            d["as_of"] = t[0][0]
            text = text[:m.start(1)] + text[m.end(1):]  # so the time stamp is not read as anything else
    now = d["as_of"] or default_as_of

    for sent in _sentences(text):
        low = sent.lower()
        times = _times(sent)

        # 2-3. server outages and late or early external events, clause by clause
        #      ("HMEM-02 is down 23:15 to 03:15, and supplier EDI won't arrive until 02:30")
        outage_sentence = bool(re.search(OUTAGE_WORDS, low))
        last_node, open_ext = None, None
        for clause in re.split(r",\s*(?:and\s+|but\s+)?|;\s*|\s+and\s+|\s+but\s+", sent):
            cl = clause.lower()
            ctimes = _times(clause)
            cnodes = sorted({f"{m.group(1).upper()}-0{m.group(2)}" for m in NODE_RE.finditer(clause)} & nodes)
            if cnodes:
                last_node = cnodes
            use_nodes = cnodes or (last_node if ctimes and re.search(OUTAGE_WORDS, cl) else None)
            if use_nodes and outage_sentence and ctimes and not _find_ext(clause):
                if len(ctimes) >= 2:
                    start, end = ctimes[0][0], ctimes[1][0]
                elif now:
                    start, end = now, ctimes[0][0]
                    notes.append(f"{use_nodes[0]} outage taken to start now ({now}) and end {end}")
                else:
                    continue
                for nid in use_nodes:
                    d["node_outages"].append({"node_id": nid, "start": start, "end": end, "reason": _reason(low)})
                continue
            ext = _find_ext(clause) or (open_ext if ctimes and re.search(DELAY_WORDS, cl) else None)
            if ext and re.search(DELAY_WORDS, cl + " " + low):
                expected = spec["external"][ext]["expected"]
                mins, dtext = _duration(clause)
                new = None
                if ctimes:
                    new = ctimes[-1][0]
                    if mins and re.search(r"\babout|around|roughly|approx", cl):
                        notes.append(f"'{dtext}' with a stated time: used the time, {new}")
                elif mins:
                    sign = -1 if re.search(r"\bearly|ahead", cl) else 1
                    new = to_hhmm(expected + sign * mins)
                    notes.append(f"'{dtext}' {'early' if sign < 0 else 'late'} read as {to_hhmm(expected)} "
                                 f"{'-' if sign < 0 else '+'} {mins} min = {new}")
                if new:
                    d["external_delays"] = [e for e in d["external_delays"] if e["ext_id"] != ext]
                    d["external_delays"].append({"ext_id": ext, "new_time": new})
                    notes.append(f"'{_alias_hit(ext, cl) if _find_ext(clause) else _alias_hit(ext, low)}' matched to "
                                 f"{ext} ({spec['external'][ext]['description']})")
                    open_ext = None
                else:
                    open_ext = ext  # time may come in the next clause: "...running late, expect it at 11pm"
        ext = _find_ext(sent)

        # 4-6. holds, priority changes and new deadlines, clause by clause
        #      ("skip X and Y", "make X P1 and Y due by 05:30")
        pending, last = [], None
        for clause in re.split(r",\s*|\s+and\s+|\s+also\s+|\s+plus\s+", sent):
            jobs = _find_jobs(spec, clause)
            intent = _intent(clause, ext)
            if intent:
                for j in pending + jobs:
                    _apply(d, spec, intent, j)
                pending, last = [], intent
            elif jobs:
                if last and last[0] == "hold":
                    for j in jobs:
                        _apply(d, spec, last, j)
                else:
                    pending += jobs
        if pending and last:
            for j in pending:
                _apply(d, spec, last, j)

        # 7. cautious runtimes
        if re.search(r"worst[- ]case|p90|cautious runtime|pessimistic|slow tonight|san is slow", low):
            d["use_p90_runtimes"] = True
            notes.append("cautious (p90) runtimes requested")

    if not d["as_of"] and default_as_of:
        notes.append(f"message gives no time; used {default_as_of} from the 'time now' field")
    d["_as_of_from_message"] = bool(d["as_of"])
    return d


def _intent(clause, ext_in_sentence):
    low = clause.lower()
    if re.search(HOLD_WORDS, low):
        return ("hold", None)
    pm = re.search(r"\bP([1-4])\b", clause)
    if pm:
        return ("priority", f"P{pm.group(1)}")
    by = re.search(r"\b(?:by|before|deadline(?: of| is| to)?|done at|finished at)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", clause, re.I)
    soft_hard = "soft" if re.search(r"\bsoft\b", low) else "hard" if re.search(r"\bhard\b", low) else None
    if by or soft_hard:
        t = _times(by.group(1)) if by else []
        return ("deadline", (t[0][0] if t else None, soft_hard))
    return None


def _apply(d, spec, intent, j):
    kind, val = intent
    name = _name(spec, j)
    if kind == "hold" and j not in d["job_holds"]:
        d["job_holds"].append(j)
        d["assumptions"].append(f"held job matched to {j} ({name})")
    elif kind == "priority":
        d["priority_changes"].append({"job_id": j, "priority": val})
        d["assumptions"].append(f"priority change matched to {j} ({name})")
    elif kind == "deadline":
        when, soft_hard = val
        job = next(x for x in spec["jobs"] if x["id"] == j)
        change = {"job_id": j, "deadline": when or to_hhmm(job["deadline"])}
        if soft_hard:
            change["deadline_type"] = soft_hard
        d["deadline_changes"].append(change)
        d["assumptions"].append(f"deadline change matched to {j} ({name})")


def _pair_after(sentence, node):
    i = sentence.upper().find(node)
    t = _times(sentence[i:])
    return (t[0][0], t[1][0]) if len(t) >= 2 else (None, None)


def _reason(low):
    for w in ("patching", "patch", "maintenance", "fan", "reboot", "upgrade", "failure", "outage"):
        if w in low:
            return w
    return "outage"


def _alias_hit(ext, low):
    for a in EXT_ALIASES[ext]:
        if a in low:
            return a
    return ext


def _name(spec, jid):
    return next(j["name"] for j in spec["jobs"] if j["id"] == jid)
