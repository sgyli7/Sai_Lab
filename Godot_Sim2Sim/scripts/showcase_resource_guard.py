"""Read-only training observation and an exclusive lease for our preview process."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
def default_training_root():
    # A separate Git worktree shares only repository metadata. Observe the
    # original checkout by default, while all output stays in this worktree.
    result = subprocess.run(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                            cwd=ROOT, capture_output=True, text=True)
    common = Path(result.stdout.strip()) if result.returncode == 0 else ROOT / '.git'
    return common.parent if common.name == '.git' else ROOT


TRAINING_ROOT = Path(os.environ.get('SIM2SIM_TRAINING_ROOT', str(default_training_root()))).resolve()
RUNS = Path(os.environ.get('SIM2SIM_TRAINING_RUNS', str(TRAINING_ROOT / 'results/research_20260910/runs')))


class PreviewDeferred(RuntimeError):
    """A resource boundary prevented rendering; this is not a renderer failure."""


def is_interactive_workshop(args):
    if '-m' not in args or args.index('-m') + 1 >= len(args):
        return False
    if args[args.index('-m') + 1] != 'sim2sim.workshop':
        return False
    return not any(arg == flag or arg.startswith(flag + '=')
                   for arg in args for flag in ('--plan', '--headless', '--record'))


def training_processes():
    """Only retain IDs/run names, never log unrelated process arguments."""
    result = []
    interactive = set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if int(proc.name) == os.getpid() or os.getpid() in _process_ancestors(int(proc.name)):
                continue
            name = (proc / 'comm').read_text().strip()
            if name not in ('python', 'python3', 'python3.12', 'godot'):
                continue
            args = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            cwd = (proc / 'cwd').resolve()
            if not (cwd == TRAINING_ROOT or TRAINING_ROOT in cwd.parents
                    or any(str(TRAINING_ROOT) in arg for arg in args)):
                continue
            run = args[args.index('--name') + 1] if '--name' in args else None
            # A user's persistent play window is not a training/evaluation job.
            # Planned, recorded and headless workshop runs remain protected.
            if is_interactive_workshop(args):
                interactive.add(int(proc.name))
            supervisor = any(arg == str(TRAINING_ROOT / 'src/sim2sim/research/budget.py') for arg in args)
            result.append({'pid': int(proc.name), 'name': name, 'run': run, 'budget_supervisor': supervisor})
        except (OSError, IndexError):
            # Process may exit between reading its fields.
            continue
    return [proc for proc in result if proc['pid'] not in interactive
            and not (_process_ancestors(proc['pid']) & interactive)]


def training_sample():
    samples = {}
    now = time.time()
    for path in RUNS.glob('*/metrics.jsonl'):
        try:
            stat = path.stat()
            if now - stat.st_mtime > 100:
                continue
            with path.open('rb') as stream:
                stream.seek(max(0, stat.st_size - 64000))
                lines = stream.read().decode(errors='replace').splitlines()
            rows = []
            for line in lines:
                try:
                    row = json.loads(line)
                    if row.get('fps', 0) > 0:
                        rows.append({'iteration': row['iteration'], 'fps': row['fps']})
                except (ValueError, KeyError):
                    continue
            if rows:
                samples[path.parent.name] = {
                    'fps': statistics.median(row['fps'] for row in rows[-8:]),
                    'iteration': rows[-1]['iteration'], 'rows': rows,
                    'age_seconds': now - stat.st_mtime,
                }
        except OSError:
            continue
    return samples


def _process_ancestors(pid):
    ancestors = set()
    for _ in range(64):
        try:
            fields = (Path('/proc') / str(pid) / 'status').read_text().splitlines()
            parent = int(next(line for line in fields if line.startswith('PPid:')).split()[1])
        except (OSError, ValueError, StopIteration):
            break
        if parent <= 1 or parent in ancestors:
            break
        ancestors.add(parent)
        pid = parent
    return ancestors


def unmeasured_processes(processes):
    """A measured trainer covers its own children and orchestration parent only.

    An unrelated evaluation/probe must not become invisible merely because a
    different trainer has fresh metrics. Read ancestry, never signal processes.
    """
    unknown = [proc for proc in processes if not proc['run']]
    if not unknown:
        return []
    measured = {proc['pid'] for proc in processes if proc['run']}
    parents = set().union(*(_process_ancestors(pid) for pid in measured)) if measured else set()
    return [proc for proc in unknown
            if proc['pid'] not in parents and not (_process_ancestors(proc['pid']) & measured)]


def pressure_sample():
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        memory[key] = int(value.strip().split()[0])
    pressures = {}
    for name in ('memory', 'cpu', 'io'):
        for line in Path(f'/proc/pressure/{name}').read_text().splitlines():
            fields = line.split()
            pressures[f'{name}_{fields[0]}_avg10'] = float(fields[1].split('=')[1])
    return {'available_memory_fraction': memory['MemAvailable'] / memory['MemTotal'], **pressures}


def pressure_reason(sample):
    if sample['available_memory_fraction'] < .10:
        return 'available memory below 10%'
    if sample['memory_full_avg10'] > 1 or sample['io_full_avg10'] > 10:
        return 'memory or I/O resource pressure'
    if sample['cpu_some_avg10'] > 80:
        return 'CPU scheduling pressure above 80%'
    return None


def competing_processes(processes, cpu_only=False):
    """Headless checks may share the host only with disjoint CPU allocations.

    Visible captures retain the original full-host exclusion. An unpinned
    evaluator intersects our allocation and therefore still blocks headless.
    """
    if not cpu_only:
        return processes
    ours = os.sched_getaffinity(0)
    result = []
    for proc in processes:
        # The repository's budget wrapper only monitors its pinned subprocess.
        # Keep that subprocess (and every worker) subject to the affinity check.
        if proc.get('budget_supervisor') and any(
                proc['pid'] in _process_ancestors(child['pid']) for child in processes if child != proc):
            continue
        try:
            if ours & os.sched_getaffinity(proc['pid']):
                result.append(proc)
        except ProcessLookupError:
            continue
    return result


def preflight(fps=30, cpu_only=False, shared_eval=False):
    processes = training_processes()
    observed = processes
    processes = competing_processes(processes, cpu_only)
    samples = training_sample()
    pressure = pressure_sample()
    reason = pressure_reason(pressure)
    if reason:
        raise PreviewDeferred(reason)
    if fps > 30 and processes:
        raise PreviewDeferred('training processes are active; tests above 30 FPS are deferred')
    if unmeasured_processes(processes) and not shared_eval:
        raise PreviewDeferred('independent training/evaluation activity has no comparable throughput; rendering deferred')
    active_runs = {p['run'] for p in processes if p['run']}
    unavailable = [name for name in active_runs
                   if name not in samples or len(samples[name]['rows']) < 8]
    if unavailable or (processes and not active_runs and not shared_eval):
        raise PreviewDeferred('training throughput cannot be compared reliably yet; rendering deferred')
    return {'time': time.time(), 'processes': processes, 'samples': samples, 'pressure': pressure,
            'cpu_only': cpu_only, 'shared_eval': shared_eval,
            'cpu_affinity': sorted(os.sched_getaffinity(0)), 'observed_processes': observed}


def _checkpoint_session_unlocked(snapshot):
    """Keep the same reference across consecutive short captures, under our lease.

    Relaunching a 55-second capture must not reset the two-window rule. A new
    training PID is a new phase; completed runs do not carry strikes forward.
    """
    path = ROOT / 'logs/session_guard.json'
    try:
        previous = json.loads(path.read_text())
    except (OSError, ValueError):
        previous = {}
    current = {}
    reasons = []
    for proc in snapshot['processes']:
        name = proc['run']
        if not name:
            continue
        sample = snapshot['samples'].get(name)
        if not sample:
            reasons.append(f'no comparable throughput for {name}')
            if previous.get(name,{}).get('pid')==proc['pid']:
                current[name]=previous[name]
            continue
        state = previous.get(name, {})
        if state.get('pid') != proc['pid']:
            state = {'pid': proc['pid'], 'reference_fps': sample['fps'],
                     'cursor': sample['iteration'], 'window_time': snapshot['time'], 'strikes': 0}
        if snapshot['time'] - state['window_time'] >= 60:
            fresh = [row['fps'] for row in sample['rows'] if row['iteration'] > state['cursor']]
            if not fresh:
                reasons.append(f'no new training iteration in the last window: {name}')
            else:
                ratio = statistics.median(fresh) / state['reference_fps']
                state['strikes'] = state['strikes'] + 1 if ratio < .95 else 0
                state['ratio'] = ratio
                state['cursor'] = sample['iteration']
                state['window_time'] = snapshot['time']
        if state['strikes'] >= 2:
            reasons.append(f'two observed windows below the original phase reference: {name}')
        current[name] = state
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(current,indent=2))
    if reasons:
        raise PreviewDeferred('; '.join(reasons))
    return current


def checkpoint_session(snapshot):
    lock=ROOT/'logs/session_guard.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:return _checkpoint_session_unlocked(snapshot)
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


class ThroughputGuard:
    """Compare distinct completed iterations in two successive 60-second windows."""
    def __init__(self, baseline):
        self.baseline = baseline['samples']
        self.cursor = {name: row['iteration'] for name, row in self.baseline.items()}
        self.strikes = {}

    def compare(self, current, active_runs):
        ratios = {}
        for name in active_runs:
            row = current.get(name)
            if not row:
                return f'no fresh throughput for active run {name}', ratios
            if name not in self.baseline:
                return f'training phase changed to {name}; recapture an idle baseline', ratios
            fresh = [x['fps'] for x in row['rows'] if x['iteration'] > self.cursor[name]]
            if not fresh:
                return f'no new completed training iteration for {name}', ratios
            ratio = statistics.median(fresh) / self.baseline[name]['fps']
            ratios[name] = ratio
            self.cursor[name] = row['iteration']
            self.strikes[name] = self.strikes.get(name, 0) + 1 if ratio < .95 else 0
            if self.strikes[name] >= 2:
                return f'training throughput down over 5% in two windows: {name}', ratios
        return None, ratios


def _watch(stop, baseline, pause):
    guard = ThroughputGuard(baseline)
    last_window = time.monotonic()
    baseline_runs={(p['run'],p['pid']) for p in baseline['processes'] if p['run']}
    while not stop.wait(5):
        pressure = pressure_sample()
        reason = pressure_reason(pressure)
        processes=competing_processes(training_processes(), baseline.get('cpu_only', False))
        active_runs={(p['run'],p['pid']) for p in processes if p['run']}
        samples=training_sample()
        if unmeasured_processes(processes) and not baseline.get('shared_eval',False):
            reason = reason or 'independent training/evaluation activity has no comparable throughput'
        if baseline.get('render_fps',30)>30 and processes:
            reason = reason or 'training became active during the independent performance test'
        if (processes and not active_runs and not baseline.get('shared_eval',False)) or any(name not in samples for name,_ in active_runs):
            reason=reason or 'the active training phase has no reliable throughput record'
        if active_runs-baseline_runs:
            reason=reason or 'training changed phase; capture a new phase baseline before rendering'
        record = {'time': time.time(), 'pressure': pressure, 'training_processes': processes}
        if time.monotonic() - last_window >= 60:
            runs = {p['run'] for p in processes if p['run']}
            reason, ratios = guard.compare(samples, runs) if not reason else (reason, {})
            if processes and not samples and not baseline.get('shared_eval',False):
                reason = reason or 'active training has no reliable throughput record'
            record.update({'samples': samples, 'ratios': ratios, 'strikes': guard.strikes})
            last_window = time.monotonic()
        record['pause'] = reason
        with (ROOT / 'logs/resource_guard.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        if reason:
            pause(reason)
            return


def watch(stop, baseline, pause):
    try:
        _watch(stop,baseline,pause)
    except Exception as error:
        pause(f'resource monitor unavailable: {type(error).__name__}: {error}')


@contextmanager
def preview_lease():
    lock = ROOT / '.cache/preview.lock'
    lock.parent.mkdir(exist_ok=True, parents=True)
    with lock.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PreviewDeferred('another Atelier preview is already running') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
