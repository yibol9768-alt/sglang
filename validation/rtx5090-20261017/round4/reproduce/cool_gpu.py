"""Wait for passive cooling; never change power, clocks, fans or other jobs."""
import time


def wait(target=55, timeout=240):
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    started = time.monotonic()
    announced = False
    while True:
        temperature = pynvml.nvmlDeviceGetTemperature(handle, 0)
        power = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
        if abs(power - 400) > 1:
            raise RuntimeError(f"Power limit is {power} W, expected 400 W")
        if temperature <= target:
            if announced:
                print(f"COOLED {temperature} C after {time.monotonic()-started:.1f}s", flush=True)
            return
        if not announced:
            print(f"COOLING {temperature} C; waiting for <= {target} C", flush=True)
            announced = True
        if time.monotonic() - started > timeout:
            raise TimeoutError("Passive cooling did not reach the test start temperature")
        time.sleep(2)


if __name__ == "__main__":
    wait()
