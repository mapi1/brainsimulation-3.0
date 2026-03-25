"""
watch_plot.py  —  live-reload wrapper
Run this once; it re-executes the target script every time you save it.

Usage:
    python watch_plot.py tvbo-network-insets.py
    python watch_plot.py          # defaults to tvbo-network-insets.py
"""
import subprocess, sys, time, os, signal

TARGET = sys.argv[1] if len(sys.argv) > 1 else "tvbo-network-insets.py"
TARGET = os.path.abspath(os.path.join(os.path.dirname(__file__), TARGET))
PYTHON = sys.executable
POLL = 0.5  # seconds between mtime checks


def launch():
    return subprocess.Popen([PYTHON, TARGET])


proc = launch()
last_mtime = os.path.getmtime(TARGET)
print(f"Watching {TARGET}  (Ctrl-C to stop)")

try:
    while True:
        time.sleep(POLL)
        try:
            mtime = os.path.getmtime(TARGET)
        except FileNotFoundError:
            continue
        if mtime != last_mtime:
            last_mtime = mtime
            print(f"\n[{time.strftime('%H:%M:%S')}] change detected – reloading…")
            # terminate old window gracefully, then force-kill if needed
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            proc = launch()
except KeyboardInterrupt:
    proc.terminate()
    print("\nstopped.")
