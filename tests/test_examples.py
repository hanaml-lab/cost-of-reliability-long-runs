import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_job_on_the_bundled_accounts_payable_job():
    out = _example("plan_job").main([str(ROOT / "examples" / "data" / "ap_job.yaml"), "--prices", "0.01", "0.2", "--runs", "500"])
    cheap, dear = out["plans"][0.01], out["plans"][0.2]
    # a higher price on tokens buys fewer checks, and less predicted success
    assert len(dear.checks) < len(cheap.checks)
    assert dear.cost < cheap.cost and dear.success <= cheap.success
