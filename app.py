"""Project Dawn: the demo app.

Type what happened tonight, press the button, and see how today's scheduler, the classical solver
and the quantum solver repair the night.

Run:  streamlit run app.py      (then open the address it prints, usually http://localhost:8501)
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from common import DATA, RESULTS  # noqa: E402
from dawn import Settings, draw_charts, run_dawn, save_result  # noqa: E402
import ibm_quantum  # noqa: E402
import watsonx_client  # noqa: E402

st.set_page_config(page_title="Project Dawn", page_icon="🌅", layout="wide")
watsonx_client._load_dotenv()

INK, CLASSICAL, QUANTUM, AMBER, MISS, OK = "#1B2A4A", "#2F5D8C", "#6B4FBB", "#E0892B", "#B3261E", "#1F7A4D"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextArea textarea, .stButton button {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; }}
code, pre, .stCode {{ font-family: 'IBM Plex Mono', ui-monospace, monospace; }}
.block-container {{ padding-top: 3.2rem; max-width: 1280px; }}
h4 {{ margin-top: 1.6rem !important; }}
.horizon {{ border-radius: 14px; padding: 26px 30px 22px; margin-bottom: 18px; color: #fff;
  background: linear-gradient(180deg, #1B2A4A 0%, #2E3F6E 46%, #7A5C8E 72%, {AMBER} 100%); }}
.horizon h1 {{ font-size: 2.3rem; font-weight: 700; margin: 0; letter-spacing: -0.01em; color: #fff; }}
.horizon p {{ margin: 6px 0 0; font-size: 1.05rem; color: #F3E6D8; max-width: 70ch; }}
.verdict {{ border-left: 6px solid; border-radius: 8px; padding: 12px 16px; background: #fff; height: 100%; }}
.verdict .who {{ font-size: 0.95rem; color: #4A5878; margin: 0; }}
.verdict .what {{ font-size: 1.35rem; font-weight: 600; margin: 4px 0 2px; color: {INK}; }}
.verdict .why {{ font-size: 0.92rem; color: #4A5878; margin: 0; }}
.side-head {{ font-size: 1.25rem; font-weight: 600; margin: 0 0 4px; }}
.brief {{ background: #fff; border-radius: 10px; padding: 16px 20px; font-size: 1.05rem; line-height: 1.55; color: {INK}; }}
</style>
<div class="horizon"><h1>Project Dawn</h1>
<p>Something went wrong in tonight's batch. Describe it below, and see how today's scheduler, a classical
optimiser and a quantum computer repair the night before morning.</p></div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ settings panel
def env(name, default=""):
    v = os.environ.get(name, "")
    return "" if v.startswith("your-") else (v or default)


REGIONS = {"Dallas (us-south)": "https://us-south.ml.cloud.ibm.com",
           "Frankfurt (eu-de)": "https://eu-de.ml.cloud.ibm.com",
           "London (eu-gb)": "https://eu-gb.ml.cloud.ibm.com",
           "Tokyo (jp-tok)": "https://jp-tok.ml.cloud.ibm.com",
           "Sydney (au-syd)": "https://au-syd.ml.cloud.ibm.com",
           "Toronto (ca-tor)": "https://ca-tor.ml.cloud.ibm.com"}

with st.sidebar:
    st.header("Settings")

    st.subheader("Reading the message")
    reader = st.radio("Message reader", ["Offline reader (no keys needed)", "watsonx.ai Granite"],
                      index=1 if watsonx_client.configured() else 0, label_visibility="collapsed")
    wx_key = wx_project = wx_url = wx_model = ""
    if reader.startswith("watsonx"):
        wx_key = st.text_input("IBM Cloud API key", value=env("WATSONX_APIKEY"), type="password")
        wx_project = st.text_input("watsonx.ai project ID", value=env("WATSONX_PROJECT_ID"))
        current = env("WATSONX_URL", REGIONS["Dallas (us-south)"])
        names = list(REGIONS)
        region = st.selectbox("Region", names, index=next((i for i, n in enumerate(names) if REGIONS[n] == current), 0))
        wx_url = REGIONS[region]
        wx_model = st.text_input("Model ID", value=env("WATSONX_MODEL_ID", "ibm/granite-4-h-small"),
                                 help="Check the exact ID in watsonx.ai Prompt Lab for your region.")
        if st.button("Test watsonx.ai connection"):
            os.environ.update(WATSONX_APIKEY=wx_key, WATSONX_PROJECT_ID=wx_project, WATSONX_URL=wx_url,
                              WATSONX_MODEL_ID=wx_model)
            try:
                with st.spinner("Calling watsonx.ai"):
                    watsonx_client.test_connection()
                st.success("Connected to watsonx.ai.")
            except Exception as e:
                st.error(f"Could not connect: {str(e)[:300]}")
    else:
        st.caption("Understands common wording: late feeds, server outages, new deadlines, skipped jobs. "
                   "Use watsonx.ai for anything more unusual.")

    st.subheader("Quantum computer")
    qc = st.radio("Quantum computer", ["Local simulator (free, about 30 s)", "IBM Quantum hardware"],
                  label_visibility="collapsed")
    ibm_token = ibm_instance = ibm_backend = ""
    if qc.startswith("IBM"):
        ibm_token = st.text_input("IBM Quantum API key", value=env("IBM_QUANTUM_TOKEN"), type="password",
                                  help="From quantum.cloud.ibm.com, on the dashboard.")
        ibm_instance = st.text_input("Instance CRN (optional)", value=env("IBM_QUANTUM_INSTANCE"),
                                     help="Leave empty to search every instance this key can use.")
        if st.button("Check IBM Quantum connection"):
            try:
                with st.spinner("Asking IBM Quantum which machines you can use"):
                    st.session_state.backends = ibm_quantum.list_backends(ibm_token, ibm_instance)
                st.success(f"Connected. {len(st.session_state.backends)} machines available.")
            except Exception as e:
                st.session_state.backends = []
                st.error(f"Could not connect: {str(e)[:300]}")
        machines = st.session_state.get("backends", [])
        labels = ["Least busy machine"] + [f"{b['name']} ({b['qubits']} qubits, {b['queue'] if b['queue'] is not None else '?'} jobs queued)"
                                           for b in machines]
        pick = st.selectbox("Machine", labels)
        ibm_backend = "" if pick == labels[0] else machines[labels.index(pick) - 1]["name"]
        st.caption("Training runs on the local simulator; only the final sampling job goes to IBM, "
                   "using a few seconds of your QPU time plus queue time.")

    with st.expander("Advanced"):
        options = st.slider("Options per hotspot job", 2, 5, 4)
        reps = st.slider("QAOA layers", 1, 4, 3)
        shots = st.select_slider("Samples (shots)", [1024, 2048, 4096, 8192], 4096)
        full = st.checkbox("Also run the full classical re-plan (reference)", True)

    if st.button("Save these keys to .env on this laptop"):
        lines = [f"WATSONX_APIKEY={wx_key or env('WATSONX_APIKEY')}", f"WATSONX_PROJECT_ID={wx_project or env('WATSONX_PROJECT_ID')}",
                 f"WATSONX_URL={wx_url or env('WATSONX_URL', REGIONS['Dallas (us-south)'])}",
                 f"WATSONX_MODEL_ID={wx_model or env('WATSONX_MODEL_ID', 'ibm/granite-4-h-small')}",
                 f"IBM_QUANTUM_TOKEN={ibm_token or env('IBM_QUANTUM_TOKEN')}",
                 f"IBM_QUANTUM_INSTANCE={ibm_instance or env('IBM_QUANTUM_INSTANCE')}"]
        (ROOT / ".env").write_text("\n".join(lines) + "\n")
        st.success("Saved to .env. It is ignored by Git; don't share it.")
    st.caption("Keys stay on this laptop. They are only sent to IBM.")


# ------------------------------------------------------------------ the message
examples = {e["title"]: e["message"] for e in json.loads((DATA / "demo_examples.json").read_text())}
for line in (DATA / "llm_test_set.jsonl").read_text().splitlines()[1:]:  # more wording to try the reader on
    if line.strip():
        m = json.loads(line)["message"]
        examples["Test message: " + m[:56] + ("..." if len(m) > 56 else "")] = m

if "message" not in st.session_state:
    st.session_state.message = examples[next(iter(examples))]


def use_example():
    st.session_state.message = examples[st.session_state.example]


left, right = st.columns([3, 1])
with left:
    st.text_area("What happened tonight?", key="message", height=130)
with right:
    st.selectbox("Or start from an example", list(examples), key="example", on_change=use_example)
    as_of = st.text_input("Time now, if the message doesn't say", "23:15")

go = st.button("Repair tonight's plan", type="primary", width="stretch")

if go:
    if reader.startswith("watsonx"):
        os.environ.update(WATSONX_APIKEY=wx_key, WATSONX_PROJECT_ID=wx_project, WATSONX_URL=wx_url,
                          WATSONX_MODEL_ID=wx_model)
    use_ibm = qc.startswith("IBM")
    if use_ibm and not ibm_quantum.has_credentials(ibm_token):
        st.error("Add your IBM Quantum API key in Settings, or switch to the local simulator.")
        st.stop()
    settings = Settings(reader="watsonx" if reader.startswith("watsonx") else "offline",
                        backend="ibm" if use_ibm else "aer", ibm_token=ibm_token, ibm_instance=ibm_instance,
                        ibm_backend=ibm_backend, default_as_of=as_of.strip() or None, options_per_job=options,
                        reps=reps, shots=shots, full_replan=full)
    with st.status("Repairing tonight's plan", expanded=True) as status:
        def progress(stage, message):
            status.update(label=message)
            if stage != "done":
                st.write(message)
        result = run_dawn(st.session_state.message, settings, progress)
        if not result["errors"]:
            status.update(label="Drawing the charts")
            stamp = time.strftime("%Y%m%d-%H%M%S")
            result["charts"] = draw_charts(result, stamp=stamp)
            save_result(result, RESULTS / f"run-{stamp}.json")
            status.update(label=f"Done in {result['timings']['total']:.0f} seconds", state="complete", expanded=False)
        else:
            status.update(label="Need more detail", state="error")
    st.session_state.result = result


# ------------------------------------------------------------------ results
r = st.session_state.get("result")
if not r:
    st.info("Describe a late feed, a server outage, a new deadline or a job to skip, then press "
            "**Repair tonight's plan**. The demo message is already filled in.")
    st.stop()

for n in r.get("notes", []):
    st.warning(n)
if r["errors"]:
    st.error("The message couldn't be turned into changes yet:\n\n" + "\n".join(f"- {e}" for e in r["errors"]))
    with st.expander("What the message reader understood"):
        st.json(r.get("disruption", {}))
    st.stop()

S = r["summaries"]
today, classical, quantum = S.get("Shift-right (today)"), S.get("Classical repair"), S.get("Quantum repair")


def miss_text(key):
    m = r["misses"].get(key, [])
    if not m:
        return "Every deadline met"
    return f"{len(m)} deadline{'s' if len(m) > 1 else ''} missed"


def miss_detail(key):
    m = r["misses"].get(key, [])
    if not m:
        return "Nothing late."
    return "; ".join(f"{x['job']} {x['minutes_late']} min late{' (hard)' if x['hard'] else ''}" for x in m[:3])


def verdict(col, who, key, color):
    col.markdown(f'<div class="verdict" style="border-color:{color}"><p class="who">{who}</p>'
                 f'<p class="what">{miss_text(key)}</p><p class="why">{miss_detail(key)}</p></div>',
                 unsafe_allow_html=True)


st.markdown(f"#### Tonight at {r['as_of']}")
c1, c2, c3 = st.columns(3)
verdict(c1, "Today's scheduler (shift the plan later)", "shift_right", MISS if r["misses"]["shift_right"] else OK)
if "classical" in r["schedules"]:
    verdict(c2, "Classical solver", "classical", CLASSICAL)
    if "quantum" in r["schedules"]:
        verdict(c3, f"Quantum solver ({r['quantum']['where']})", "quantum", QUANTUM)
    elif r.get("quantum", {}).get("error"):
        c3.error(f"Quantum run failed: {r['quantum']['error']}")
    else:
        c3.info("Quantum solver found no valid repair this run. Try more samples or fewer options per job.")
else:
    c2.info(r["notes"][-1] if r["notes"] else "No hotspot to solve.")

if "hotspot" in r:
    h = r["hotspot"]
    st.markdown(f"#### The contested decision: {len(h['jobs'])} jobs fighting over scarce servers and licences")
    st.caption(f"{h['combinations']} possible combinations, {h['conflicts']} clashing pairs. "
               f"The quantum solver uses one qubit per option: {h['qubits']} qubits.")

    def cell(p):
        return "" if not p else f"{p['node']} {p['start']} to {p['end']}, {p['status']}"

    table = pd.DataFrame([{"Job": d["job"], "Name": d["name"],
                           "Deadline": d["deadline"] + (" (hard)" if d["hard"] else ""),
                           "Today's scheduler": cell(d["today"]), "Classical": cell(d["classical"]),
                           "Quantum": cell(d["quantum"])} for d in r["decisions"]])
    st.dataframe(table, hide_index=True, width="stretch")

    cl, qu = st.columns(2)
    with cl:
        c = r["classical"]
        st.markdown(f'<p class="side-head" style="color:{CLASSICAL}">Classical solver</p>', unsafe_allow_html=True)
        st.markdown(f"Checked all {h['combinations']} combinations in {c['seconds']:.2f} s.  \n"
                    f"Best hotspot cost **{c['cost']:g}**, against {c['shift_right_cost']:g} for today's scheduler.")
    with qu:
        q = r.get("quantum", {})
        st.markdown(f'<p class="side-head" style="color:{QUANTUM}">Quantum solver (QAOA)</p>', unsafe_allow_html=True)
        if q.get("error"):
            st.error(q["error"])
        elif q:
            same = "the same answer as the classical solver" if q.get("matches_classical") else "a worse answer than the classical solver"
            lines = [f"{q['qubits']} qubits, {q['reps']} layers, {q['shots']:,} samples on the {q['where']}.",
                     f"Best hotspot cost **{q['cost']:g}**: {same}." if q.get("cost") is not None else "No valid repair sampled.",
                     f"Valid repairs in **{q['valid_share']:.1%}** of samples, against {q['random_valid_share']:.1%} for random guessing.",
                     f"Time: {q['train_seconds']:.0f} s training on the simulator, {q['sample_seconds']:.1f} s sampling."]
            if q.get("job_id"):
                lines.append(f"IBM job {q['job_id']} on {q['backend']}" + (f", {q['qpu_seconds']} s of QPU time." if q.get("qpu_seconds") else "."))
            st.markdown("  \n".join(lines))
        else:
            st.caption("Quantum solver was not run.")

st.markdown("#### The night, before and after")
names = {"plan": "Published plan", "shift_right": "Today's scheduler", "classical": "Classical repair", "quantum": "Quantum repair",
         "full_replan": "Full re-plan (reference)"}
charts = r.get("charts", {})
if charts:
    tabs = st.tabs([names[k] for k in charts])
    for tab, (k, path) in zip(tabs, charts.items()):
        with tab:
            st.image(path, width="stretch")

st.markdown("#### Briefing for the night-shift lead")
st.markdown(f'<div class="brief">{r["briefing"]}</div>', unsafe_allow_html=True)

st.markdown("#### All approaches compared")
rows = []
for name, s in S.items():
    if name == "Published plan":
        continue
    rows.append({"Approach": name, "Deadlines missed": s["sla_misses"], "Hard deadlines missed": s["hard_misses"],
                 "Minutes late": s["minutes_late"], "Jobs moved": s["jobs_moved"], "Batch ends": s["batch_finishes"],
                 "Score (lower is better)": s["score"]})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with st.expander("What the message reader understood"):
    st.caption(f"Reader used: {'watsonx.ai Granite' if r['reader_used'] == 'watsonx' else 'offline reader'}")
    for a in r["disruption"].get("assumptions", []):
        st.markdown(f"- {a}")
    st.json({k: v for k, v in r["disruption"].items() if k != "assumptions"})
if "hotspot" in r:
    with st.expander("The quantum sub-problem: every option the solvers chose between"):
        for j, opts in r["hotspot"]["options"].items():
            st.markdown(f"**{j}**: " + "; ".join(f"{o['node']} {o['start']} to {o['end']} (cost {o['cost']:g})" for o in opts))
with st.expander("Timings"):
    st.json(r["timings"])
st.download_button("Download this run (JSON)", json.dumps({k: v for k, v in r.items() if not k.startswith("_")},
                   indent=2, default=list), file_name="project-dawn-run.json")
