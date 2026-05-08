"""
data_collector.py
Script untuk mengumpulkan data latency jaringan secara otomatis.
Menjalankan ping ke server target dan menyimpan hasilnya ke CSV.

Usage:
  python data_collector.py                  # Collect 1 sample (untuk GitHub Actions cron)
  python data_collector.py batch 100        # Collect 100 samples cepat (untuk bootstrap lokal)
"""

import subprocess
import platform
import re
import csv
import os
import sys
import time
from datetime import datetime


def ping_server(host, count=10):
    """Melakukan ping ke server dan mengekstrak metrik."""
    system = platform.system().lower()

    if system == "windows":
        cmd = ["ping", "-n", str(count), host]
    else:
        cmd = ["ping", "-c", str(count), host]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout

        if system == "windows":
            return _parse_windows(output)
        else:
            return _parse_linux(output)
    except Exception as e:
        print(f"  Error pinging {host}: {e}")
        return {"latency_ms": -1, "jitter": -1, "packet_loss": 100.0}


def _parse_windows(output):
    times = [float(t) for t in re.findall(r'time[=<](\d+)ms', output)]
    loss_match = re.search(r'\((\d+)% loss\)', output)
    packet_loss = float(loss_match.group(1)) if loss_match else 100.0

    if times:
        avg = sum(times) / len(times)
        jitter = sum(abs(times[i+1] - times[i]) for i in range(len(times)-1)) / max(len(times)-1, 1)
    else:
        avg, jitter = -1, -1

    return {"latency_ms": round(avg, 2), "jitter": round(jitter, 2), "packet_loss": packet_loss}


def _parse_linux(output):
    times = [float(t) for t in re.findall(r'time=(\d+\.?\d*)\s*ms', output)]
    loss_match = re.search(r'(\d+)% packet loss', output)
    packet_loss = float(loss_match.group(1)) if loss_match else 100.0

    if times:
        avg = sum(times) / len(times)
        jitter = sum(abs(times[i+1] - times[i]) for i in range(len(times)-1)) / max(len(times)-1, 1)
    else:
        avg, jitter = -1, -1

    return {"latency_ms": round(avg, 2), "jitter": round(jitter, 2), "packet_loss": packet_loss}


def collect_one_sample(output_path="latency_raw.csv", hosts=None):
    """Collect 1 sample dari semua host. Untuk GitHub Actions cron."""
    if hosts is None:
        hosts = ["8.8.8.8", "1.1.1.1", "8.8.4.4"]

    file_exists = os.path.exists(output_path)
    fieldnames = [
        "timestamp", "server", "day_of_week", "hour", "minute",
        "latency_ms", "jitter", "packet_loss", "is_weekend"
    ]

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for host in hosts:
            now = datetime.now()
            metrics = ping_server(host, count=5)

            if metrics["latency_ms"] <= 0:
                print(f"  SKIP {host}: no response")
                continue

            row = {
                "timestamp": now.isoformat(),
                "server": host,
                "day_of_week": now.weekday(),
                "hour": now.hour,
                "minute": now.minute,
                "latency_ms": metrics["latency_ms"],
                "jitter": metrics["jitter"],
                "packet_loss": metrics["packet_loss"],
                "is_weekend": 1 if now.weekday() >= 5 else 0
            }
            writer.writerow(row)
            print(f"  [{now.strftime('%H:%M:%S')}] {host}: "
                  f"lat={metrics['latency_ms']}ms, jit={metrics['jitter']}ms, "
                  f"loss={metrics['packet_loss']}%")

    print(f"Data saved to {output_path}")


def collect_batch(output_path="latency_raw.csv", n_rounds=30, delay=10):
    """
    Collect banyak sample secara cepat dengan jeda pendek.
    n_rounds=30, delay=10 -> ~100 rows dalam ~5 menit.
    """
    hosts = ["8.8.8.8", "1.1.1.1", "8.8.4.4"]
    print(f"Collecting {n_rounds} rounds x {len(hosts)} hosts = ~{n_rounds * len(hosts)} rows")
    print(f"Delay between rounds: {delay}s")
    print(f"Estimated time: ~{n_rounds * delay // 60} minutes\n")

    for i in range(n_rounds):
        print(f"--- Round {i+1}/{n_rounds} ---")
        collect_one_sample(output_path, hosts)
        if i < n_rounds - 1:
            time.sleep(delay)

    # Count rows
    with open(output_path, "r") as f:
        rows = sum(1 for _ in f) - 1  # minus header
    print(f"\nDone! Total rows in {output_path}: {rows}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        collect_batch("latency_raw.csv", n_rounds=n, delay=10)
    else:
        collect_one_sample("latency_raw.csv")
