#!/usr/bin/env bash
# Sequential Godot/Jolt fine-tunes for the 8 non-walking demo policies.
# exec train_walk_godot.sh so SIGTERM reaches Python. Do not `uv sync`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
export SIM2SIM_ROOT="$ROOT"
export MICRODUCK_POLICIES="${MICRODUCK_POLICIES:-$HOME/Projects/MicroDuck/policies}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export SIM2SIM_TORCH_THREADS="${SIM2SIM_TORCH_THREADS:-2}"

TRAIN="$ROOT/scripts/train_walk_godot.sh"
if [[ ! -x "$TRAIN" ]]; then
  echo "missing $TRAIN" >&2
  exit 1
fi

# name config [extra args...]
run_one() {
  local name="$1"
  local cfg="$2"
  shift 2
  echo "== train $name ($cfg) ==" >&2
  "$TRAIN" --config "$cfg" "$@"
}

# Override with SKILL=stand to run one. FORCE=1 retrains even if export exists.
SKILL="${SKILL:-all}"

should_run() {
  local name="$1"
  local onnx="$2"
  if [[ "$SKILL" != "all" && "$SKILL" != "$name" ]]; then
    return 1
  fi
  if [[ "${FORCE:-0}" != "1" && -f "$onnx" ]]; then
    echo "== skip $name: $onnx exists (FORCE=1 to retrain) ==" >&2
    return 1
  fi
  return 0
}

if should_run stand "$MICRODUCK_POLICIES/Stand_Godot.onnx"; then
  run_one stand configs/stand_godot.yaml
fi
if should_run sitstand "$MICRODUCK_POLICIES/Sitstand_Godot.onnx"; then
  run_one sitstand configs/sitstand_godot.yaml
fi
if should_run pick "$MICRODUCK_POLICIES/GroundPick_Godot.onnx"; then
  run_one pick configs/pick_godot.yaml
fi
if should_run kick_left "$MICRODUCK_POLICIES/KickLeft_Godot.onnx"; then
  run_one kick_left configs/kick_left_godot.yaml
fi
if should_run kick_right "$MICRODUCK_POLICIES/KickRight_Godot.onnx"; then
  run_one kick_right configs/kick_right_godot.yaml
fi
if should_run roulade "$MICRODUCK_POLICIES/Roulade_Godot.onnx"; then
  run_one roulade configs/roulade_godot.yaml
fi
if should_run roller "$MICRODUCK_POLICIES/Roller_Godot.onnx"; then
  run_one roller configs/roller_godot.yaml
fi
if should_run roller_crouch "$MICRODUCK_POLICIES/RollerCrouch_Godot.onnx"; then
  run_one roller_crouch configs/roller_crouch_godot.yaml
fi

echo "== skill train script done ==" >&2
