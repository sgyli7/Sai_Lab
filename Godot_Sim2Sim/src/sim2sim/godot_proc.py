"""Launch a Godot scene as a lockstep physics server."""

from __future__ import annotations

import ctypes
import fcntl
import itertools
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from sim2sim.paths import sim2sim_root
from sim2sim.protocol import JsonLineClient, wait_connect


GODOT_PROJECT = sim2sim_root() / "godot"
_CORE_SEQ = itertools.count()


def _headless_core() -> int:
    # A supervisor/taskset restriction must survive spawning physics workers.
    # On unrestricted hosts retain the original two-core reservation.
    allowed=sorted(os.sched_getaffinity(0))
    pool=allowed[:-2] if len(allowed)>4 else allowed
    return pool[next(_CORE_SEQ)%len(pool)]


def godot_bin() -> str:
    return os.environ.get("GODOT") or os.path.expanduser("~/.local/bin/godot")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _headless_overlay(src: Path) -> Path:
    """Temp --path with override.cfg capping WorkerThreadPool at 1 thread.

    Each stock Godot process otherwise starts nproc WorkerThreads (20 on this
    box). 16 workers × 20 threads saturates the scheduler; lockstep has one
    Jolt island and does not benefit from that pool.
    """
    overlay = Path(tempfile.mkdtemp(prefix="godot-s2s-ov-"))
    try:
        for entry in src.iterdir():
            if entry.name == "override.cfg":
                continue
            dest = overlay / entry.name
            # Copy project.godot so Godot does not canonicalize --path through a
            # symlink and miss overlay/override.cfg.
            if entry.name == "project.godot" and entry.is_file():
                shutil.copy2(entry, dest)
            else:
                os.symlink(entry, dest)
        (overlay / "override.cfg").write_text(
            "; sim2sim headless lockstep: 1 robot per process\n"
            "[threading]\n"
            "worker_pool/max_threads=1\n"
        )
    except Exception:
        shutil.rmtree(overlay, ignore_errors=True)
        raise
    return overlay


PR_SET_PDEATHSIG = 1


def _set_pdeathsig() -> None:
    """Linux: Godot dies with SIGTERM when the Python parent exits (no orphans)."""
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGTERM))
    except (OSError, AttributeError):
        return
    # Race: parent already gone before prctl.
    try:
        if os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGTERM)
    except OSError:
        pass


def _child_preexec(core: int | None):
    def _inner() -> None:
        _set_pdeathsig()
        if core is None:
            return
        try:
            os.sched_setaffinity(0, {int(core)})
        except (AttributeError, OSError):
            pass

    return _inner


def _cleanup_overlay(proc: subprocess.Popen) -> None:
    overlay = getattr(proc, "_sim2sim_overlay", None)
    if overlay is None:
        return
    try:
        shutil.rmtree(overlay, ignore_errors=True)
    except Exception:
        pass
    proc._sim2sim_overlay = None  # type: ignore[attr-defined]


def spawn_godot(
    scene: str,
    *,
    port: int | None = None,
    headless: bool = True,
    extra_args: list[str] | None = None,
    cwd: Path | None = None,
    recv_timeout: float = 120.0,
) -> tuple[subprocess.Popen, int, JsonLineClient]:
    # Choosing an unused port and releasing its probe socket is not a
    # reservation. Concurrent trials could select the same port before either
    # Godot process listened, then connect to each other's simulation.
    lock_path=Path(tempfile.gettempdir())/f'godot-sim2sim-spawn-{os.getuid()}.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _spawn_godot_locked(scene,port=port,headless=headless,
            extra_args=extra_args,cwd=cwd,recv_timeout=recv_timeout)


def _spawn_godot_locked(
    scene: str,
    *,
    port: int | None = None,
    headless: bool = True,
    extra_args: list[str] | None = None,
    cwd: Path | None = None,
    recv_timeout: float = 120.0,
) -> tuple[subprocess.Popen, int, JsonLineClient]:
    port = port or free_port()
    bin_ = godot_bin()
    project = Path(cwd or GODOT_PROJECT)
    overlay: Path | None = None
    if headless:
        try:
            overlay = _headless_overlay(project)
            project = overlay
        except OSError:
            overlay = None
    cmd = [bin_]
    # Optional fallback when Vulkan device creation fails (e.g. flaky GPU).
    driver = os.environ.get("GODOT_RENDERING_DRIVER", "").strip()
    if driver:
        cmd += ["--rendering-driver", driver]
    elif not headless and os.environ.get("SIM2SIM_FORCE_GL"):
        # GB10/Spark: Vulkan device creation fails (-3); OpenGL works.
        cmd += ["--rendering-driver", "opengl3"]
    if headless:
        cmd.append("--headless")
        # --fixed-fps disables wall-clock pacing (still 1 physics tick / frame
        # at physics_ticks_per_second=200). Observed >>200 ticks/s wall.
        cmd += ["--fixed-fps", "200", "--max-fps", "0", "--single-threaded-scene"]
    else:
        # 200 Hz main loop so 4 lockstep ticks are not bound to 60 Hz vsync.
        cmd += ["--fixed-fps", "200"]
        if os.environ.get("SIM2SIM_DISABLE_VSYNC", "0") == "1":
            # Only add this where the GL driver tolerates it (validated on
            # NVIDIA 580.xx/aarch64: the flag segfaults Godot at GL init, so
            # it must be OFF by default).
            cmd += ["--disable-vsync"]
        # On a Wayland-capable desktop, Godot prefers Wayland even when an
        # X11 DISPLAY is set; the screenshot/window tooling here is X11.
        if os.environ.get("SIM2SIM_DISPLAY_DRIVER"):
            cmd += ["--display-driver", os.environ["SIM2SIM_DISPLAY_DRIVER"]]
    cmd += [
        "--path",
        str(project),
        scene,
        "--",
        f"--port={port}",
    ]
    if extra_args:
        cmd += extra_args
    env = os.environ.copy()
    if env.get("DISPLAY") and not env.get("XAUTHORITY"):
        for cand in (
            Path.home() / ".Xauthority",
            Path(f"/run/user/{os.getuid()}/gdm/Xauthority"),
        ):
            if cand.is_file():
                env["XAUTHORITY"] = str(cand)
                break
    log_file=tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',
        prefix=f'godot-sim2sim-{port}-',suffix='.log',delete=False)
    log_path=Path(log_file.name)
    core: int | None = None
    if headless:
        core = _headless_core()
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        preexec_fn=_child_preexec(core),
    )
    proc._sim2sim_log_path = log_path  # type: ignore[attr-defined]
    proc._sim2sim_log_file = log_file  # type: ignore[attr-defined]
    proc._sim2sim_overlay = overlay  # type: ignore[attr-defined]
    try:
        # Only connect after this exact child reports ownership of its socket.
        # A foreign listener on an explicitly occupied port must not receive a
        # research command or be closed during cleanup.
        deadline=time.monotonic()+25.
        marker=f'sim2sim_physics_server listening 127.0.0.1:{port}'
        while marker not in log_path.read_text(encoding='utf-8',errors='replace'):
            if proc.poll() is not None or time.monotonic()>=deadline:
                raise RuntimeError('The spawned Godot process did not acquire its listener')
            time.sleep(.01)
        client = wait_connect("127.0.0.1", port, timeout=25.0, recv_timeout=recv_timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
        try:
            log_file.close()
        except Exception:
            pass
        _cleanup_overlay(proc)
        out = ""
        try:
            out = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"Godot failed to accept TCP on {port}.\ncmd={cmd}\nlog:\n{out}") from None
    return proc, port, client


def stop_godot(proc: subprocess.Popen, client: JsonLineClient | None) -> str:
    if client is not None:
        try:
            client.call({"cmd": "close"})
        except Exception:
            pass
        try:
            client.close()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    log_file = getattr(proc, "_sim2sim_log_file", None)
    if log_file is not None:
        try:
            log_file.close()
        except Exception:
            pass
    _cleanup_overlay(proc)
    log_path = getattr(proc, "_sim2sim_log_path", None)
    if log_path is not None:
        try:
            return Path(log_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
    return ""
