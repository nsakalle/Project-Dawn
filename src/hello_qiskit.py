"""First check: build a 2-qubit Bell state and measure it.

Expected result: roughly half '00' and half '11', and never '01' or '10'.
On real hardware you will see a few '01'/'10' counts: that is noise.

Run (simulator):   python src/hello_qiskit.py
Run (hardware):    python src/hello_qiskit.py --ibm      (uses a few seconds of your free QPU time)
"""
import sys

from qiskit import QuantumCircuit, transpile

qc = QuantumCircuit(2)
qc.h(0)        # put qubit 0 in superposition
qc.cx(0, 1)    # entangle qubit 1 with qubit 0
qc.measure_all()
print(qc.draw())

if "--ibm" in sys.argv:
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2
    from ibm_quantum import get_service

    service = get_service()  # IBM_QUANTUM_TOKEN in .env, or an account saved with save_account
    print("Backends you can use:", [b.name for b in service.backends(simulator=False, operational=True)])
    backend = service.least_busy(operational=True, simulator=False)
    isa = generate_preset_pass_manager(optimization_level=1, backend=backend).run(qc)
    job = SamplerV2(mode=backend).run([isa], shots=1000)
    print(f"Job {job.job_id()} sent to {backend.name}, waiting...")
    counts = job.result()[0].data.meas.get_counts()
else:
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    counts = sim.run(transpile(qc, sim), shots=1000).result().get_counts()

print("Counts:", dict(sorted(counts.items())))
