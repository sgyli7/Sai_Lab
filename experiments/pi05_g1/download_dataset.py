#!/usr/bin/env python3
"""Download Arena-G1 LeRobot assets at a pinned Hugging Face revision.

The script is idempotent and resumable. It copies same-revision files from the
local source tree when the content hash matches, and otherwise downloads them
into ``.scratch/pi05_g1/data``. The duplicate HDF5 recording is not downloaded.
At most four downloads run at once. Account, gate, and quota failures stop the
run; incomplete files stay on disk so a later run can continue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.hf_api import RepoFile

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "nvidia/Arena-G1-Static-PickNPlace-Task"
REVISION = "37ba80a99486a4c308477854f54b4e82e77777dc"
DEFAULT_DEST = REPO_ROOT / ".scratch" / "pi05_g1" / "data"
DEFAULT_CACHE = REPO_ROOT / ".scratch" / "pi05_g1" / "hf_cache"
DEFAULT_REUSE = Path(
    "/home/ethan/models/unitree_g1/static_source_expert/37ba80a99486a4c308477854f54b4e82e77777dc"
)
MAX_WORKERS = 4
ETAG_TIMEOUT_S = 30.0
DOWNLOAD_ATTEMPTS = 3
WANTED_SUFFIXES = (".parquet", ".mp4", ".json", ".jsonl")


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    blob_id: str
    lfs_sha256: str | None

    @property
    def content_sha256(self) -> str | None:
        return self.lfs_sha256


def wanted(path: str) -> bool:
    if path.endswith(".hdf5"):
        return False
    if path == "README.md":
        return True
    if not path.startswith("lerobot/"):
        return False
    return path.endswith(WANTED_SUFFIXES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_matches(path: Path, remote: RemoteFile) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size != remote.size:
        return False
    if remote.lfs_sha256 is not None:
        return sha256_file(path) == remote.lfs_sha256
    return git_blob_sha1(path) == remote.blob_id


def is_account_failure(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status in {401, 403, 429}:
        return True
    text = str(exc).lower()
    markers = (
        "quota",
        "gated",
        "unauthorized",
        "forbidden",
        "rate limit",
        "too many requests",
        "insufficient",
    )
    return any(marker in text for marker in markers)


def list_remote_files() -> list[RemoteFile]:
    api = HfApi()
    entries = api.list_repo_tree(
        REPO_ID,
        revision=REVISION,
        repo_type="dataset",
        recursive=True,
        expand=True,
    )
    files: list[RemoteFile] = []
    for entry in entries:
        if not isinstance(entry, RepoFile):
            continue
        if not wanted(entry.path):
            continue
        lfs_sha256 = entry.lfs.sha256 if entry.lfs is not None else None
        files.append(
            RemoteFile(
                path=entry.path,
                size=int(entry.size),
                blob_id=entry.blob_id,
                lfs_sha256=lfs_sha256,
            )
        )
    files.sort(key=lambda item: item.path)
    return files


def copy_reuse(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".copying")
    with src.open("rb") as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    os.replace(temporary, dest)


def download_one(
    remote: RemoteFile,
    dest_root: Path,
    cache_dir: Path,
    reuse_root: Path | None,
    stop: threading.Event,
) -> dict[str, object]:
    if stop.is_set():
        return {"path": remote.path, "status": "skipped_after_stop", "bytes": 0}
    dest = dest_root / remote.path
    started = time.perf_counter()
    if file_matches(dest, remote):
        logging.info("skip existing %s bytes=%s", remote.path, remote.size)
        return {"path": remote.path, "status": "skipped", "bytes": remote.size, "seconds": 0.0}

    if reuse_root is not None:
        source = reuse_root / remote.path
        if source.is_file() and not source.is_symlink() and file_matches(source, remote):
            copy_reuse(source, dest)
            if not file_matches(dest, remote):
                dest.unlink(missing_ok=True)
                raise RuntimeError(f"copied file failed verification: {remote.path}")
            logging.info("copied reuse %s bytes=%s", remote.path, remote.size)
            return {
                "path": remote.path,
                "status": "copied",
                "bytes": remote.size,
                "seconds": round(time.perf_counter() - started, 3),
            }

    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        if stop.is_set():
            return {"path": remote.path, "status": "skipped_after_stop", "bytes": 0}
        try:
            logging.info("download attempt %s/%s %s", attempt, DOWNLOAD_ATTEMPTS, remote.path)
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                revision=REVISION,
                filename=remote.path,
                local_dir=dest_root,
                cache_dir=cache_dir,
                force_download=dest.is_file(),
                etag_timeout=ETAG_TIMEOUT_S,
            )
            if not file_matches(dest, remote):
                raise RuntimeError(f"downloaded file failed verification: {remote.path}")
            logging.info("downloaded %s bytes=%s", remote.path, remote.size)
            return {
                "path": remote.path,
                "status": "downloaded",
                "bytes": remote.size,
                "seconds": round(time.perf_counter() - started, 3),
            }
        except Exception as exc:
            last_error = exc
            if is_account_failure(exc):
                stop.set()
                logging.error("account or quota failure on %s: %s", remote.path, exc)
                return {
                    "path": remote.path,
                    "status": "account_failure",
                    "bytes": 0,
                    "error": str(exc),
                }
            logging.warning("download failed %s attempt %s: %s", remote.path, attempt, exc)
            time.sleep(min(2 * attempt, 8))
    return {
        "path": remote.path,
        "status": "failed",
        "bytes": 0,
        "error": str(last_error),
        "error_type": type(last_error).__name__ if last_error else "unknown",
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--reuse-root", type=Path, default=DEFAULT_REUSE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dest_root = args.dest.resolve()
    cache_dir = args.cache.resolve()
    reuse_root = args.reuse_root.resolve() if args.reuse_root else None
    state_dir = dest_root / "_download_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(state_dir / "download.log"),
        ],
    )
    logging.info(
        "repo=%s revision=%s dest=%s reuse=%s workers=%s etag_timeout=%s download_timeout=%s",
        REPO_ID,
        REVISION,
        dest_root,
        reuse_root,
        MAX_WORKERS,
        ETAG_TIMEOUT_S,
        os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT"),
    )
    if reuse_root is not None and not reuse_root.is_dir():
        logging.warning("reuse root is missing; downloads will not copy local files: %s", reuse_root)
        reuse_root = None

    try:
        remote_files = list_remote_files()
    except HfHubHTTPError as exc:
        logging.error("listing repository failed: %s", exc)
        if is_account_failure(exc):
            return 3
        return 2
    except Exception as exc:
        logging.error("listing repository failed: %s", exc)
        return 2

    write_json(
        state_dir / "remote_manifest.json",
        {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "file_count": len(remote_files),
            "bytes": sum(item.size for item in remote_files),
            "files": [item.__dict__ for item in remote_files],
        },
    )
    logging.info(
        "selected %s files, %s bytes; excluded hdf5 and non-lerobot extras",
        len(remote_files),
        sum(item.size for item in remote_files),
    )

    stop = threading.Event()
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(download_one, remote, dest_root, cache_dir, reuse_root, stop)
            for remote in remote_files
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result.get("status") == "account_failure":
                stop.set()
                for pending in futures:
                    pending.cancel()

    results.sort(key=lambda item: str(item["path"]))
    counts: dict[str, int] = {}
    byte_counts: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        counts[status] = counts.get(status, 0) + 1
        byte_counts[status] = byte_counts.get(status, 0) + int(result.get("bytes") or 0)
    present = sum(1 for remote in remote_files if file_matches(dest_root / remote.path, remote))
    summary = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "dest": str(dest_root),
        "selected_files": len(remote_files),
        "selected_bytes": sum(item.size for item in remote_files),
        "verified_files": present,
        "counts": counts,
        "bytes_by_status": byte_counts,
        "results": results,
    }
    write_json(state_dir / "receipt.json", summary)
    logging.info(
        "done selected=%s verified=%s counts=%s bytes=%s",
        len(remote_files),
        present,
        counts,
        byte_counts,
    )
    if any(result.get("status") == "account_failure" for result in results):
        return 3
    if present != len(remote_files):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
