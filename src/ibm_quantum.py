"""Connecting to IBM Quantum with an API key.

Three ways to give the key, checked in this order:
  1. passed in directly (the app's settings panel does this)
  2. IBM_QUANTUM_TOKEN / IBM_QUANTUM_INSTANCE in the .env file
  3. an account saved once with QiskitRuntimeService.save_account(...)

The API key comes from https://quantum.cloud.ibm.com (dashboard, "API key").
The instance is the CRN of your Open Plan instance; it can be left empty, and then every instance
the key can see is searched.
"""
import os

CHANNEL = "ibm_quantum_platform"


def _env(name):
    from watsonx_client import _load_dotenv  # reads .env into os.environ (without overriding)
    _load_dotenv()
    v = os.environ.get(name, "").strip()
    return v if v and not v.startswith("your-") else None


def get_service(token=None, instance=None):
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = (token or "").strip() or _env("IBM_QUANTUM_TOKEN")
    instance = (instance or "").strip() or _env("IBM_QUANTUM_INSTANCE")
    if token:
        return QiskitRuntimeService(channel=CHANNEL, token=token, instance=instance)
    return QiskitRuntimeService()  # falls back to the saved account; raises a clear error if none


def has_credentials(token=None):
    if (token or "").strip() or _env("IBM_QUANTUM_TOKEN"):
        return True
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        return bool(QiskitRuntimeService.saved_accounts())
    except Exception:
        return False


def list_backends(token=None, instance=None):
    """Real quantum computers this key can use, with qubit counts and queue lengths."""
    service = get_service(token, instance)
    out = []
    for b in service.backends(simulator=False, operational=True):
        try:
            pending = b.status().pending_jobs
        except Exception:
            pending = None
        out.append({"name": b.name, "qubits": b.num_qubits, "queue": pending})
    return sorted(out, key=lambda r: (r["queue"] is None, r["queue"] or 0))
