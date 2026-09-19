#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/showcase_env.sh"
SHOWCASE_CPUS="$(.venv/bin/python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[-2:])))')"
exec nice -n 10 taskset -c "$SHOWCASE_CPUS" .venv/bin/python scripts/play_atelier.py "$@"
