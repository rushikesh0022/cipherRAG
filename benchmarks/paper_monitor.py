import os
import platform
import threading
import time

import psutil


def process_tree_rss_mb(process: psutil.Process) -> float:
    total = 0
    try:
        total += process.memory_info().rss
    except Exception:
        pass

    try:
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except Exception:
                pass
    except Exception:
        pass

    return total / (1024 * 1024)


def get_system_info() -> dict:
    vm = psutil.virtual_memory()
    return {
        "hostname": platform.node(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_mb": round(vm.total / (1024 * 1024), 2),
    }


class ProcessMonitor:
    """
    Monitors current Python process tree (including children).
    """

    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.process = psutil.Process(os.getpid())
        self.running = False
        self.thread = None
        self.samples = []

    def _sample_loop(self):
        while self.running:
            try:
                rss_mb = process_tree_rss_mb(self.process)
                cpu_percent = self.process.cpu_percent(interval=None)
                vm = psutil.virtual_memory()
                self.samples.append(
                    {
                        "ts": time.time(),
                        "rss_mb": rss_mb,
                        "cpu_percent": cpu_percent,
                        "system_mem_used_mb": vm.used / (1024 * 1024),
                        "system_mem_percent": vm.percent,
                    }
                )
            except Exception:
                pass
            time.sleep(self.sample_interval)

    def start(self):
        self.samples = []
        self.running = True
        self.process.cpu_percent(interval=None)
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def summary(self) -> dict:
        if not self.samples:
            return {
                "peak_rss_mb": 0.0,
                "avg_rss_mb": 0.0,
                "peak_cpu_percent": 0.0,
                "avg_cpu_percent": 0.0,
                "peak_system_mem_used_mb": 0.0,
                "avg_system_mem_percent": 0.0,
            }

        peak_rss = max(s["rss_mb"] for s in self.samples)
        avg_rss = sum(s["rss_mb"] for s in self.samples) / len(self.samples)
        peak_cpu = max(s["cpu_percent"] for s in self.samples)
        avg_cpu = sum(s["cpu_percent"] for s in self.samples) / len(self.samples)
        peak_sys_mem = max(s["system_mem_used_mb"] for s in self.samples)
        avg_sys_mem_pct = sum(s["system_mem_percent"] for s in self.samples) / len(self.samples)

        return {
            "peak_rss_mb": round(peak_rss, 2),
            "avg_rss_mb": round(avg_rss, 2),
            "peak_cpu_percent": round(peak_cpu, 2),
            "avg_cpu_percent": round(avg_cpu, 2),
            "peak_system_mem_used_mb": round(peak_sys_mem, 2),
            "avg_system_mem_percent": round(avg_sys_mem_pct, 2),
        }
