#!/usr/bin/env bash
# Closed-loop acceptance for Godot walk finetune (pipeline, not gait quality).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export SIM2SIM_ROOT="$ROOT"

if [[ -z "${MICRODUCK_POLICIES:-}" ]]; then
  for d in "$ROOT/policies" "$ROOT/../policies" "$HOME/Projects/MicroDuck/policies"; do
    if [[ -d "$d" ]]; then
      export MICRODUCK_POLICIES="$(cd "$d" && pwd)"
      break
    fi
  done
fi
ALPHA="${MICRODUCK_POLICIES:-}/alpha_walking.onnx"
if [[ ! -f "$ALPHA" ]]; then
  echo "missing alpha_walking.onnx (MICRODUCK_POLICIES=${MICRODUCK_POLICIES:-unset})" >&2
  exit 1
fi

OUT="${WALK_GODOT_SMOKE_OUT:-/tmp/walk_godot_smoke}"
rm -rf "$OUT"
mkdir -p "$OUT"
LOG_ROOT="$OUT/logs"

UV=(uv run --no-sync)
if "${UV[@]}" sim2sim-train --help >/dev/null 2>&1; then
  TRAIN=("${UV[@]}" sim2sim-train)
else
  TRAIN=("${UV[@]}" python -m sim2sim.train.runner)
fi

echo "== (a) train 3 iters from alpha ONNX =="
"${TRAIN[@]}" \
  --config configs/walk_godot.yaml \
  --init-onnx alpha \
  --num-envs 2 \
  --samples-per-iter 48 \
  --critic-warmup-iters 1 \
  --max-iterations 3 \
  --run-name smoke \
  --device cpu \
  --seed 0 \
  --log-root "$LOG_ROOT"

RUN_DIR="$(ls -td "$LOG_ROOT"/*_smoke | head -1)"
test -n "$RUN_DIR"
test -d "$RUN_DIR"
CKPT=""
for cand in "$RUN_DIR/model_2.pt" "$RUN_DIR/model_1.pt" "$RUN_DIR/model_0.pt"; do
  if [[ -f "$cand" ]]; then
    CKPT="$cand"
    break
  fi
done
test -f "$CKPT"
test -f "$RUN_DIR/params/init_check.json"
"${UV[@]}" python - <<PY
import json
from pathlib import Path
p = Path("$RUN_DIR") / "params" / "init_check.json"
d = json.loads(p.read_text())
err = float(d["max_abs_err"])
assert err < 1e-5, d
assert d["ok"] is True, d
print(f"init_check ok max_abs_err={err:.6e}")
PY

echo "== (b) resume 1 iter =="
"${TRAIN[@]}" \
  --config configs/walk_godot.yaml \
  --resume "$CKPT" \
  --num-envs 2 \
  --samples-per-iter 48 \
  --critic-warmup-iters 0 \
  --max-iterations 1 \
  --run-name smoke_resume \
  --device cpu \
  --seed 0 \
  --log-root "$LOG_ROOT"

RESUME_DIR="$(ls -td "$LOG_ROOT"/*_smoke_resume | head -1)"
RESUME_CKPT="$RESUME_DIR/model_3.pt"
if [[ ! -f "$RESUME_CKPT" ]]; then
  RESUME_CKPT="$(ls -1 "$RESUME_DIR"/model_*.pt | sort -V | tail -1)"
fi
test -f "$RESUME_CKPT"
"${UV[@]}" python - <<PY
import torch
from pathlib import Path
old = torch.load("$CKPT", map_location="cpu", weights_only=False)
new = torch.load("$RESUME_CKPT", map_location="cpu", weights_only=False)
print(f"resume iter {old['iter']} -> {new['iter']} file=$RESUME_CKPT")
assert int(new["iter"]) > int(old["iter"]), (old["iter"], new["iter"])
PY

ONNX="$OUT/Walk_Godot.onnx"
echo "== (c) export $ONNX =="
"${UV[@]}" sim2sim-export --checkpoint "$RESUME_CKPT" --out "$ONNX"
test -f "$ONNX"
test -f "${ONNX%.onnx}.manifest.json" || test -f "$OUT/Walk_Godot.manifest.json"

echo "== (d) godot runner rollout =="
NPZ="$OUT/rollout.npz"
"${UV[@]}" sim2sim-runner --backend godot --onnx "$ONNX" --out "$NPZ"
test -f "$NPZ"

echo "== (e) play bank accepts exported walking ONNX =="
"${UV[@]}" python - <<PY
from pathlib import Path
from sim2sim.play import load_bank, policy_paths
onnx = Path("$ONNX")
paths = policy_paths(local_ppo=False, walking=onnx)
bank = load_bank(paths, 14)
assert "walking" in bank, list(bank)
print("play bank walking=", bank["walking"].path)
PY

echo "== (f) eval_walk A/B smoke =="
"${UV[@]}" sim2sim-eval-walk \
  --a "$ALPHA" \
  --b "$ONNX" \
  --seeds 1 \
  --seconds 3 \
  --workers 2 \
  --out "$OUT/eval"

test -f "$OUT/eval/metrics.json"
test -f "$OUT/eval/report.md"

echo "WALK_GODOT_SMOKE: PASS"
