"""Optional GPU guard: commands are wrapped only when GPU_GUARD is set."""

from __future__ import annotations

import sys

from ir_agent.gpu import guard_script, guarded, warn_if_unguarded


def test_guard_unset_runs_command_directly(monkeypatch):
    monkeypatch.delenv("GPU_GUARD", raising=False)
    cmd = ["python", "-m", "ir_agent", "validate"]
    assert guard_script() is None
    assert guarded(cmd, 3.0, "note") == cmd


def test_guard_set_wraps_command(monkeypatch, tmp_path):
    script = tmp_path / "gpu_guard.py"
    monkeypatch.setenv("GPU_GUARD", str(script))
    cmd = ["python", "x.py", "--inner"]
    out = guarded(cmd, "3.0", "Run C")
    assert out[:3] == [sys.executable, str(script), "run"]
    assert out[out.index("--gb") + 1] == "3.0"
    assert out[out.index("--note") + 1] == "Run C"
    assert out[out.index("--") + 1:] == cmd


def test_no_warning_without_guard(monkeypatch, caplog):
    monkeypatch.delenv("GPU_GUARD", raising=False)
    monkeypatch.delenv("GPU_GUARD_GB", raising=False)
    warn_if_unguarded("cuda")
    assert "GPU_GUARD" not in caplog.text
