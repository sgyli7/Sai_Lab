"""Typed access to the pinned MicroDuck upstream lock file."""

from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path


class LockFileError(ValueError):
    """Raised when an upstream lock file does not match the supported schema."""


@dataclass(frozen=True, slots=True)
class RepositoryLock:
    """A repository URL pinned to one immutable Git commit."""

    url: str
    commit: str


@dataclass(frozen=True, slots=True)
class UpstreamLock:
    """Pinned repositories and the policy files expected from them."""

    schema_version: int
    repositories: dict[str, RepositoryLock]
    policies: tuple[str, ...]


def load_upstream_lock(path: str | Path) -> UpstreamLock:
    """Load and validate the version-one upstream lock document."""

    lock_path = Path(path)
    try:
        raw = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LockFileError(f"Unable to read upstream lock file {lock_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise LockFileError("Upstream lock root must be a JSON object")

    schema_version = raw.get("schemaVersion")
    if schema_version != 1:
        raise LockFileError(f"Unsupported upstream lock schemaVersion: {schema_version!r}")

    raw_repositories = raw.get("repositories")
    if not isinstance(raw_repositories, dict):
        raise LockFileError("repositories must be a JSON object")

    repositories: dict[str, RepositoryLock] = {}
    for name, raw_repository in raw_repositories.items():
        if not isinstance(name, str) or not isinstance(raw_repository, dict):
            raise LockFileError("Each repository must be a named JSON object")
        url = raw_repository.get("url")
        commit = raw_repository.get("commit")
        if not isinstance(url, str) or not url:
            raise LockFileError(f"Repository {name!r} must define a non-empty URL")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in string.hexdigits for character in commit)
        ):
            raise LockFileError(
                f"Repository {name!r} commit must be a 40-character commit SHA"
            )
        repositories[name] = RepositoryLock(url=url, commit=commit.lower())

    raw_policies = raw.get("policies")
    if not isinstance(raw_policies, list) or not all(
        isinstance(policy, str) and policy for policy in raw_policies
    ):
        raise LockFileError("policies must be a list of non-empty file names")

    return UpstreamLock(
        schema_version=schema_version,
        repositories=repositories,
        policies=tuple(raw_policies),
    )
