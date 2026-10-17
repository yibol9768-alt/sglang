"""Run only our test process group, retaining the user's 400 W / 75 C policy."""
import argparse
import atexit
import json
import os
from pathlib import Path
import signal
import select
import subprocess
import sys
import time

def monitor():
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    while True:
        try:
            point = {"temperature_c": float(pynvml.nvmlDeviceGetTemperature(handle, 0)),
                     "power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000,
                     "memory_used_mib": pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2,
                     "measured_monotonic": time.monotonic()}
            print(json.dumps(point), flush=True)
        except Exception as error:
            print(json.dumps({"error": str(error)}), flush=True)
            return
        time.sleep(1)

class DeviceMonitor:
    def __init__(self):
        self.process = subprocess.Popen([sys.executable, __file__, "--monitor"], stdout=subprocess.PIPE, bufsize=0)
        self.buffer = b""
        atexit.register(self.close)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdout.close()

    def reading(self):
        deadline = time.monotonic() + 8
        while b"\n" not in self.buffer:
            ready, _, _ = select.select([self.process.stdout], [], [], max(0, deadline-time.monotonic()))
            if not ready:
                raise TimeoutError("Persistent NVML monitor has not reported for 8 seconds")
            chunk = os.read(self.process.stdout.fileno(), 8192)
            if not chunk:
                raise RuntimeError("Persistent NVML monitor exited")
            self.buffer += chunk
        while select.select([self.process.stdout], [], [], 0)[0]:
            chunk = os.read(self.process.stdout.fileno(), 8192)
            if not chunk:
                break
            self.buffer += chunk
        lines = self.buffer.split(b"\n")
        self.buffer = lines.pop()
        point = json.loads(lines[-1])
        if "error" in point:
            raise RuntimeError(point["error"])
        if time.monotonic()-point["measured_monotonic"] > 4:
            raise RuntimeError("GPU temperature reading is stale")
        return point

def stop(process, *, immediate=False):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL if immediate else signal.SIGINT)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    device_monitor = DeviceMonitor()
    first = device_monitor.reading()
    if first["temperature_c"] >= 72 or abs(first["power_limit_w"] - 400) > 1:
        raise RuntimeError(f"Test preconditions are not met: {first}")
    print("GPU guard:", json.dumps(first), flush=True)
    start = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    samples = [first]
    interrupted = None
    while process.poll() is None:
        time.sleep(1)
        try:
            point = device_monitor.reading()
        except Exception as error:
            interrupted = f"GPU monitoring unavailable: {error}"
            stop(process, immediate=True)
            break
        point["elapsed_s"] = round(time.monotonic() - start, 2)
        samples.append(point)
        if point["temperature_c"] >= 72:
            interrupted = "temperature reached 72 C"
        elif abs(point["power_limit_w"] - 400) > 1:
            interrupted = "power limit changed from 400 W"
        elif time.monotonic() - start > args.timeout:
            interrupted = "test timeout"
        if interrupted:
            stop(process, immediate=point["temperature_c"] >= 72 or abs(point["power_limit_w"] - 400) > 1)
            break
    result = {"max_temperature_c": max(s["temperature_c"] for s in samples), "power_limit_w": 400, "exit_code": process.wait(), "interrupted": interrupted, "samples": samples}
    Path(args.log).write_text(json.dumps(result, indent=2))
    print("GPU guard result:", json.dumps({k:v for k,v in result.items() if k != "samples"}), flush=True)
    raise SystemExit(1 if interrupted else result["exit_code"])

if __name__ == "__main__":
    if sys.argv[1:] == ["--monitor"]:
        monitor()
    else:
        main()
