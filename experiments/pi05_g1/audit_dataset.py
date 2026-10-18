#!/usr/bin/env python3
"""Read-only audit of the pinned Arena-G1 LeRobot checkout.

Reports trajectory and video counts, parquet schema and row lengths, video frame
counts against timestamps, missing language metadata, and the README-200 versus
info.json-251 contradiction. Writes a hash manifest and a JSON report. Does not
create splits or rewrite dataset files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / ".scratch" / "pi05_g1" / "data"
DEFAULT_REPORT_DIR = REPO_ROOT / ".scratch" / "pi05_g1" / "reports"
REVISION = "37ba80a99486a4c308477854f54b4e82e77777dc"
README_TRAJECTORIES = 200
README_VIDEOS = 200
INFO_EPISODES = 251
FPS = 50.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_boxes(data: bytes, start: int, end: int):
    position = start
    while position + 8 <= end:
        size = int.from_bytes(data[position : position + 4], "big")
        box_type = data[position + 4 : position + 8]
        header = 8
        if size == 1:
            if position + 16 > end:
                return
            size = int.from_bytes(data[position + 8 : position + 16], "big")
            header = 16
        elif size == 0:
            size = end - position
        if size < header or position + size > end:
            return
        yield box_type, position + header, position + size
        position += size


def find_child(data: bytes, start: int, end: int, box_type: bytes) -> tuple[int, int] | None:
    for current_type, child_start, child_end in iter_boxes(data, start, end):
        if current_type == box_type:
            return child_start, child_end
    return None


def parse_mvhd(payload: bytes) -> tuple[int, int] | None:
    if len(payload) < 20:
        return None
    version = payload[0]
    if version == 0 and len(payload) >= 20:
        timescale = int.from_bytes(payload[12:16], "big")
        duration = int.from_bytes(payload[16:20], "big")
        return timescale, duration
    if version == 1 and len(payload) >= 32:
        timescale = int.from_bytes(payload[20:24], "big")
        duration = int.from_bytes(payload[24:32], "big")
        return timescale, duration
    return None


def parse_hdlr(payload: bytes) -> bytes | None:
    if len(payload) < 12:
        return None
    return payload[8:12]


def parse_stts_count(payload: bytes) -> int | None:
    if len(payload) < 8:
        return None
    entry_count = int.from_bytes(payload[4:8], "big")
    needed = 8 + entry_count * 8
    if len(payload) < needed:
        return None
    total = 0
    offset = 8
    for _ in range(entry_count):
        total += int.from_bytes(payload[offset : offset + 4], "big")
        offset += 8
    return total


def video_sample_count(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    moov = find_child(data, 0, len(data), b"moov")
    if moov is None:
        return {"ok": False, "error": "moov box not found"}
    mvhd = find_child(data, moov[0], moov[1], b"mvhd")
    movie = parse_mvhd(data[mvhd[0] : mvhd[1]]) if mvhd else None
    frames = None
    track_duration_s = None
    for trak_start, trak_end in (
        (start, end) for box_type, start, end in iter_boxes(data, moov[0], moov[1]) if box_type == b"trak"
    ):
        mdia = find_child(data, trak_start, trak_end, b"mdia")
        if mdia is None:
            continue
        hdlr = find_child(data, mdia[0], mdia[1], b"hdlr")
        if hdlr is None or parse_hdlr(data[hdlr[0] : hdlr[1]]) != b"vide":
            continue
        mdhd = find_child(data, mdia[0], mdia[1], b"mdhd")
        parsed_mdhd = parse_mvhd(data[mdhd[0] : mdhd[1]]) if mdhd else None
        if parsed_mdhd and parsed_mdhd[0]:
            track_duration_s = parsed_mdhd[1] / parsed_mdhd[0]
        minf = find_child(data, mdia[0], mdia[1], b"minf")
        stbl = find_child(data, minf[0], minf[1], b"stbl") if minf else None
        stts = find_child(data, stbl[0], stbl[1], b"stts") if stbl else None
        if stts is not None:
            frames = parse_stts_count(data[stts[0] : stts[1]])
        break
    duration_s = None
    if movie and movie[0]:
        duration_s = movie[1] / movie[0]
    if frames is None:
        return {"ok": False, "error": "video sample count not found", "duration_s": duration_s}
    return {
        "ok": True,
        "frames": frames,
        "duration_s": duration_s,
        "track_duration_s": track_duration_s,
    }


def flatten_numeric(column: pa.ChunkedArray) -> list[float] | list[int] | None:
    current = column
    while pa.types.is_list(current.type) or pa.types.is_large_list(current.type) or pa.types.is_fixed_size_list(current.type):
        current = current.flatten()
    if not (
        pa.types.is_integer(current.type)
        or pa.types.is_floating(current.type)
        or pa.types.is_boolean(current.type)
    ):
        return None
    values = current.to_pylist()
    if pa.types.is_floating(current.type):
        return [None if value is None else float(value) for value in values]
    return [None if value is None else int(value) for value in values]


def schema_summary(schema: pa.Schema) -> list[dict[str, str]]:
    return [{"name": field.name, "type": str(field.type), "nullable": str(field.nullable)} for field in schema]


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def episode_index_from_name(path: Path) -> int | None:
    stem = path.stem
    prefix = "episode_"
    if not stem.startswith(prefix) or not stem[len(prefix) :].isdigit():
        return None
    return int(stem[len(prefix) :])


def readme_claims(text: str) -> dict[str, int | None]:
    trajectories = README_TRAJECTORIES if "200" in text and "Trajectories" in text else None
    videos = README_VIDEOS if "Number of RGB videos: 200" in text else None
    return {"trajectories": trajectories, "videos": videos}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def audit(data_root: Path, report_dir: Path) -> dict[str, object]:
    dataset_files = [
        path
        for path in data_root.rglob("*")
        if path.is_file() and "_download_state" not in path.relative_to(data_root).parts
    ]
    dataset_files.sort()
    hash_rows = []
    for path in dataset_files:
        relative = path.relative_to(data_root).as_posix()
        hash_rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    parquet_paths = sorted(data_root.glob("lerobot/data/chunk-*/*.parquet"))
    video_paths = sorted(data_root.glob("lerobot/videos/chunk-*/*/*.mp4"))
    info_path = data_root / "lerobot" / "meta" / "info.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    episodes = load_jsonl(data_root / "lerobot" / "meta" / "episodes.jsonl")
    tasks = load_jsonl(data_root / "lerobot" / "meta" / "tasks.jsonl")
    readme_path = data_root / "README.md"
    readme = readme_path.read_text() if readme_path.is_file() else ""

    episode_meta = {}
    for row in episodes:
        if "episode_index" in row:
            episode_meta[int(row["episode_index"])] = row
    task_by_index = {}
    for row in tasks:
        if "task_index" in row:
            task_by_index[int(row["task_index"])] = row.get("task")

    schemas: dict[str, list[dict[str, str]]] = {}
    episodes_out = []
    mismatches = []
    timestamp_issues = []
    length_mismatches = []
    missing_language = []
    bad_task_index = []
    schema_groups: dict[str, int] = {}
    total_rows = 0

    videos_by_index = {episode_index_from_name(path): path for path in video_paths}
    parquets_by_index = {episode_index_from_name(path): path for path in parquet_paths}
    indices = sorted(index for index in set(parquets_by_index) | set(videos_by_index) if index is not None)

    for index in indices:
        parquet_path = parquets_by_index.get(index)
        video_path = videos_by_index.get(index)
        record: dict[str, object] = {"episode_index": index}
        if parquet_path is None:
            record["parquet_missing"] = True
            mismatches.append({"episode_index": index, "reason": "parquet missing"})
        else:
            parquet = pq.ParquetFile(parquet_path)
            schema_key = str(parquet.schema_arrow)
            schemas.setdefault(schema_key, schema_summary(parquet.schema_arrow))
            schema_groups[schema_key] = schema_groups.get(schema_key, 0) + 1
            rows = parquet.metadata.num_rows
            total_rows += rows
            table = pq.read_table(
                parquet_path,
                columns=[
                    name
                    for name in (
                        "timestamp",
                        "annotation.human.task_description",
                        "episode_index",
                        "frame_index",
                        "task_index",
                    )
                    if name in parquet.schema_arrow.names
                ],
            )
            timestamps = flatten_numeric(table.column("timestamp")) if "timestamp" in table.column_names else None
            annotations = (
                flatten_numeric(table.column("annotation.human.task_description"))
                if "annotation.human.task_description" in table.column_names
                else None
            )
            episode_ids = flatten_numeric(table.column("episode_index")) if "episode_index" in table.column_names else None
            frame_ids = flatten_numeric(table.column("frame_index")) if "frame_index" in table.column_names else None
            record.update(
                {
                    "rows": rows,
                    "parquet": parquet_path.relative_to(data_root).as_posix(),
                    "timestamp_count": None if timestamps is None else len(timestamps),
                }
            )
            if timestamps is None or any(value is None for value in timestamps):
                timestamp_issues.append({"episode_index": index, "reason": "timestamp missing"})
            else:
                deltas = [right - left for left, right in zip(timestamps, timestamps[1:])]
                record["timestamp_start"] = timestamps[0]
                record["timestamp_end"] = timestamps[-1]
                record["timestamp_monotonic"] = all(delta >= -1e-9 for delta in deltas)
                if deltas:
                    record["timestamp_median_dt"] = statistics.median(deltas)
                if not record["timestamp_monotonic"]:
                    timestamp_issues.append({"episode_index": index, "reason": "timestamp not monotonic"})
                if len(timestamps) != rows:
                    timestamp_issues.append({"episode_index": index, "reason": "timestamp count != rows"})
            if episode_ids is not None and any(value != index for value in episode_ids):
                mismatches.append({"episode_index": index, "reason": "episode_index column disagrees with filename"})
            if frame_ids is not None and frame_ids != list(range(len(frame_ids))):
                mismatches.append({"episode_index": index, "reason": "frame_index is not 0..n-1"})
            unique_annotations = sorted({int(value) for value in annotations}) if annotations else []
            record["annotation_task_indices"] = unique_annotations
            if annotations is None:
                missing_language.append({"episode_index": index, "reason": "annotation.human.task_description column missing"})
            elif any(value not in task_by_index for value in unique_annotations):
                bad_task_index.append({"episode_index": index, "indices": unique_annotations})
        if video_path is None:
            record["video_missing"] = True
            mismatches.append({"episode_index": index, "reason": "video missing"})
        else:
            video_info = video_sample_count(video_path)
            record["video"] = video_path.relative_to(data_root).as_posix()
            record["video_info"] = video_info
            if video_info.get("ok") and parquet_path is not None and video_info.get("frames") != record.get("rows"):
                mismatches.append(
                    {
                        "episode_index": index,
                        "reason": "video frames != parquet rows",
                        "frames": video_info.get("frames"),
                        "rows": record.get("rows"),
                    }
                )
            timestamps_ok = record.get("timestamp_start") is not None and record.get("timestamp_end") is not None
            if video_info.get("ok") and timestamps_ok and record.get("rows"):
                span = float(record["timestamp_end"]) - float(record["timestamp_start"])
                expected = max(int(record["rows"]) - 1, 0) / FPS
                record["timestamp_span_s"] = span
                record["expected_span_s"] = expected
                if abs(span - expected) > (1.5 / FPS):
                    timestamp_issues.append(
                        {
                            "episode_index": index,
                            "reason": "timestamp span disagrees with (rows-1)/fps",
                            "span_s": span,
                            "expected_s": expected,
                        }
                    )
                duration = video_info.get("duration_s")
                if isinstance(duration, float) and abs(duration - (int(record["rows"]) / FPS)) > (1.5 / FPS):
                    timestamp_issues.append(
                        {
                            "episode_index": index,
                            "reason": "video duration disagrees with rows/fps",
                            "duration_s": duration,
                            "expected_s": int(record["rows"]) / FPS,
                        }
                    )
        meta = episode_meta.get(index)
        if meta is None:
            missing_language.append({"episode_index": index, "reason": "episodes.jsonl entry missing"})
        else:
            record["meta_length"] = meta.get("length")
            tasks_field = meta.get("tasks")
            record["meta_tasks"] = tasks_field
            if not tasks_field:
                missing_language.append({"episode_index": index, "reason": "episodes.jsonl tasks empty"})
            if meta.get("length") != record.get("rows") and record.get("rows") is not None:
                length_mismatches.append(
                    {"episode_index": index, "meta_length": meta.get("length"), "rows": record.get("rows")}
                )
        episodes_out.append(record)

    present_indices = set(indices)
    claimed_count = info.get("total_episodes") if isinstance(info.get("total_episodes"), int) else INFO_EPISODES
    missing_in_251 = [index for index in range(claimed_count) if index not in present_indices]
    info_episodes = info.get("total_episodes")
    info_videos = info.get("total_videos")
    info_frames = info.get("total_frames")
    report = {
        "revision": REVISION,
        "dataset_root": str(data_root),
        "readme_claims": readme_claims(readme),
        "info_json": {
            "total_episodes": info_episodes,
            "total_videos": info_videos,
            "total_frames": info_frames,
            "total_chunks": info.get("total_chunks"),
            "chunks_size": info.get("chunks_size"),
            "fps": info.get("fps"),
            "splits": info.get("splits"),
            "codebase_version": info.get("codebase_version"),
        },
        "actual": {
            "files": len(hash_rows),
            "bytes": sum(row["bytes"] for row in hash_rows),
            "parquet_files": len(parquet_paths),
            "video_files": len(video_paths),
            "episode_indices_present": indices,
            "total_rows": total_rows,
            "schema_variants": len(schema_groups),
            "schema": next(iter(schemas.values()), []),
        },
        "contradiction_200_vs_251": {
            "readme_trajectories": README_TRAJECTORIES if "200" in readme else None,
            "readme_videos": README_VIDEOS if "Number of RGB videos: 200" in readme else None,
            "info_total_episodes": info_episodes,
            "info_total_videos": info_videos,
            "info_total_frames": info_frames,
            "parquet_files": len(parquet_paths),
            "video_files": len(video_paths),
            "episodes_jsonl_records": len(episodes),
            "tasks_jsonl_records": len(tasks),
            "missing_indices_in_0_250": missing_in_251,
            "missing_index_count": len(missing_in_251),
            "sum_parquet_rows": total_rows,
            "frames_minus_rows": None if info_frames is None else info_frames - total_rows,
            "notes": [
                "README quantification says 200 demonstrations and 200 RGB videos.",
                "lerobot/meta/info.json says total_episodes 251 and total_videos 251.",
                "Filenames run through episode_000250, which is a 251-wide index space, but some indices have no parquet or video.",
                "info.json splits.train is '0:100' and total_chunks is 0; those fields are recorded and are not used as a training split.",
            ],
        },
        "alignment": {
            "episodes_checked": len(episodes_out),
            "row_video_or_index_mismatches": mismatches,
            "timestamp_issues": timestamp_issues,
            "episodes_jsonl_length_mismatches": length_mismatches,
        },
        "language": {
            "tasks_jsonl_present": (data_root / "lerobot" / "meta" / "tasks.jsonl").is_file(),
            "episodes_jsonl_present": (data_root / "lerobot" / "meta" / "episodes.jsonl").is_file(),
            "tasks": tasks,
            "episodes_missing_language": missing_language,
            "annotation_indices_missing_from_tasks": bad_task_index,
        },
        "episodes": episodes_out,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    hash_path = report_dir / "file_hashes.jsonl"
    temporary_hash = hash_path.with_suffix(".jsonl.tmp")
    with temporary_hash.open("w") as handle:
        for row in hash_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary_hash, hash_path)
    report["hash_manifest"] = str(hash_path)
    report["hash_manifest_sha256"] = sha256_file(hash_path)
    write_json(report_dir / "audit_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit(args.data.resolve(), args.report_dir.resolve())
    contradiction = report["contradiction_200_vs_251"]
    print(
        json.dumps(
            {
                "parquet_files": report["actual"]["parquet_files"],
                "video_files": report["actual"]["video_files"],
                "total_rows": report["actual"]["total_rows"],
                "readme_trajectories": contradiction["readme_trajectories"],
                "info_total_episodes": contradiction["info_total_episodes"],
                "missing_index_count": contradiction["missing_index_count"],
                "alignment_mismatches": len(report["alignment"]["row_video_or_index_mismatches"]),
                "timestamp_issues": len(report["alignment"]["timestamp_issues"]),
                "missing_language": len(report["language"]["episodes_missing_language"]),
                "report": str(args.report_dir.resolve() / "audit_report.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
