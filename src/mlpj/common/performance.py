"""可选分段计时、CUDA事件和只读硬件采样；不改变仿真时间步。"""
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import csv
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import threading
import time


def summary(values):
    values = sorted(values)
    if not values:
        return {'count': 0, 'total_seconds': 0.0}
    position = (len(values) - 1) * .95
    lo = int(position)
    p95 = values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)
    return {'count': len(values), 'total_seconds': sum(values),
            'mean_seconds': statistics.mean(values), 'median_seconds': statistics.median(values),
            'p95_seconds': p95}


class Timings:
    def __init__(self, enabled=True, cuda_events=False):
        self.enabled = enabled
        self.cuda_events = cuda_events
        self.wall = defaultdict(list)
        self.events = defaultdict(list)

    @contextmanager
    def stage(self, name, gpu=False):
        if not self.enabled:
            yield
            return
        start_event = end_event = None
        if gpu and self.cuda_events:
            import torch
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.wall[name].append(time.perf_counter() - start)
            if start_event is not None:
                end_event.record()
                self.events[name].append((start_event, end_event))

    def report(self):
        result = {'wall_stages': {k: summary(v) for k, v in self.wall.items()},
                  'wall_note': 'GPU阶段wall时间含提交/等待，异步设备时间单列；阶段可能重叠，不相加当总耗时。'}
        if self.events:
            import torch
            torch.cuda.synchronize()
            result['cuda_stages'] = {k: summary([a.elapsed_time(b) / 1000 for a, b in pairs])
                                     for k, pairs in self.events.items()}
        return result


def runtime_info():
    import torch
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
            'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'rss_note': 'Linux进程峰值；不含其他worker，GPU显存也不含OpenGL等外部分配'}


class HardwareMonitor:
    """采样系统CPU/内存和nvidia-smi；不支持字段留空并记录错误。"""
    def __init__(self, output, enabled=False, interval=1.0):
        self.output, self.enabled, self.interval = Path(output), enabled, interval
        self.stop = threading.Event()
        self.errors = set()

    def __enter__(self):
        if self.enabled:
            self.started = time.perf_counter()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self

    def _run(self):
        fields = ['elapsed_seconds', 'cpu_busy_percent', 'mem_available_bytes', 'swap_used_bytes',
                  'gpu_index', 'gpu_util_percent', 'gpu_memory_used_mib', 'gpu_temperature_c', 'gpu_power_w',
                  'cpu_per_core_busy_percent_json']
        previous = None
        previous_cores = None
        with self.output.open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            while not self.stop.is_set():
                row = {'elapsed_seconds': time.perf_counter() - self.started}
                try:
                    cpu_lines = [line.split() for line in Path('/proc/stat').read_text().splitlines() if line.startswith('cpu')]
                    values = [int(v) for v in cpu_lines[0][1:9]]
                    cores = [(sum(int(v) for v in line[1:9]), int(line[4])+int(line[5])) for line in cpu_lines[1:]]
                    if previous_cores and len(cores) == len(previous_cores):
                        row['cpu_per_core_busy_percent_json'] = json.dumps([
                            100*(1-(b-d)/(a-c)) if a>c else 0.0
                            for (a,b),(c,d) in zip(cores, previous_cores)])
                    previous_cores = cores
                    current = (sum(values), values[3] + values[4])
                    if previous and current[0] > previous[0]:
                        row['cpu_busy_percent'] = 100 * (1 - (current[1]-previous[1])/(current[0]-previous[0]))
                    previous = current
                    mem = {line.split(':')[0]: int(line.split()[1])*1024
                           for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split()) >= 2}
                    row.update(mem_available_bytes=mem.get('MemAvailable'),
                               swap_used_bytes=mem.get('SwapTotal', 0)-mem.get('SwapFree', 0))
                except (OSError, ValueError) as exc:
                    self.errors.add(str(exc))
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw',
                                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
                    if result.returncode:
                        raise RuntimeError(result.stderr.strip())
                    lines = result.stdout.strip().splitlines()
                    for line in lines:
                        values = next(csv.reader([line], skipinitialspace=True))
                        writer.writerow({**row, **dict(zip(fields[4:], values))})
                    if not lines:
                        writer.writerow(row)
                except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                    self.errors.add(str(exc))
                    writer.writerow(row)
                f.flush()
                self.stop.wait(self.interval)

    def __exit__(self, *exc):
        if self.enabled:
            self.stop.set()
            self.thread.join()
            if self.errors:
                self.output.with_suffix('.errors.json').write_text(json.dumps(sorted(self.errors), indent=2))


def precision_context(precision):
    import torch
    if precision == 'fp32':
        return nullcontext()
    return torch.autocast('cuda', dtype=torch.float16 if precision == 'amp_fp16' else torch.bfloat16)


def profiler_context(config, output):
    import torch
    if not config['profiler_enabled']:
        return nullcontext()
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=config['profiler_active_steps'], repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(Path(output)/'profiler')),
        record_shapes=False, profile_memory=True)
