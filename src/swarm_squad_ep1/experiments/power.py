"""
GPU power sampling with nvidia-smi, for per-query inference energy.
"""

import shutil
import subprocess
import threading
from datetime import datetime
from time import monotonic, sleep

import numpy as np


def query_gpu() -> dict:
    """GPU name, driver, and power limit, or {} without an NVIDIA GPU."""
    if not shutil.which("nvidia-smi"):
        return {}
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,power.limit",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return {"error": out.stdout.strip() or out.stderr.strip()}
    fields = [s.strip() for s in out.stdout.splitlines()[0].split(",")]
    if len(fields) != 4:
        return {"raw": out.stdout.strip()}
    name, driver, memory, limit = fields
    return {"name": name, "driver": driver, "memory": memory, "power_limit": limit}


class PowerMonitor:
    """
    Samples board power every interval_ms in a background thread.

    The field is power.draw.instant when the driver supports it (otherwise
    power.draw, which the driver may average over up to 1 s).
    """

    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.samples = []  # (unix time, watts)
        self.field = None
        self._proc = None
        self._thread = None
        self._lock = threading.Lock()

    def _pick_field(self):
        for field in ("power.draw.instant", "power.draw"):
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            if out.returncode == 0:
                try:
                    float(out.stdout.strip().splitlines()[0])
                    return field
                except ValueError:
                    continue
        raise RuntimeError("nvidia-smi cannot report GPU power")

    def start(self):
        self.field = self._pick_field()
        line_buffered = ["stdbuf", "-oL"] if shutil.which("stdbuf") else []
        self._proc = subprocess.Popen(
            [
                *line_buffered,
                "nvidia-smi",
                f"--query-gpu=timestamp,{self.field}",
                "--format=csv,noheader,nounits",
                f"--loop-ms={self.interval_ms}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self):
        # nvidia-smi may block-buffer its output into the pipe, so each sample
        # uses nvidia-smi's own timestamp rather than the time it is read
        for line in self._proc.stdout:
            sample = parse_sample(line)
            if sample is not None:
                with self._lock:
                    self.samples.append(sample)

    def stop(self):
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        if self._thread:
            self._thread.join(timeout=5)

    def sync(self, t, timeout=30.0) -> bool:
        """Wait until samples up to time t have been read."""
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            with self._lock:
                if self.samples and self.samples[-1][0] >= t:
                    return True
            sleep(0.05)
        return False

    def window(self, t0, t1):
        with self._lock:
            data = np.array(self.samples, dtype=float)
        if data.size == 0:
            return np.empty((0, 2))
        return data[(data[:, 0] >= t0) & (data[:, 0] <= t1)]

    def energy(self, t0, t1, idle_w=0.0) -> float:
        """Energy in joules over [t0, t1], above idle_w, by the trapezoid rule."""
        return integrate_energy(self.window(t0 - 1.0, t1 + 1.0), t0, t1, idle_w)

    def mean_power(self, t0, t1) -> float:
        data = self.window(t0, t1)
        return float(data[:, 1].mean()) if len(data) else float("nan")


def parse_sample(line: str):
    """(unix time, watts) from a "YYYY/MM/DD HH:MM:SS.fff, watts" line, or None."""
    try:
        stamp, watts = (s.strip() for s in line.split(","))
        t = datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f").timestamp()
        return t, float(watts)
    except ValueError:
        return None


def integrate_energy(samples, t0, t1, idle_w=0.0) -> float:
    """
    Integrate (power - idle_w) over [t0, t1] from (time, watts) samples,
    interpolating linearly at the window edges.
    """
    samples = np.asarray(samples, dtype=float)
    if len(samples) < 2 or t1 <= t0:
        return float("nan")
    t, p = samples[:, 0], samples[:, 1] - idle_w
    inner = (t > t0) & (t < t1)
    times = np.concatenate([[t0], t[inner], [t1]])
    power = np.interp(times, t, p)
    return float(np.sum(0.5 * (power[1:] + power[:-1]) * np.diff(times)))
