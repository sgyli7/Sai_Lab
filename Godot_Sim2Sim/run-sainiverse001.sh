#!/usr/bin/env bash
set -euo pipefail
SAINIVERSE_GAME="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SAINIVERSE_GAME/run-sainiverse-v0.1.sh" "$@"
