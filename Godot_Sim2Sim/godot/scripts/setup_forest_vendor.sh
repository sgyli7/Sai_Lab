#!/usr/bin/env bash
# Clone / refresh GamesNotDeveloped Procedural Forest Demo and wire res:// symlinks
# for MicroDuck's rough_forest_play bypass scene.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_DIR="$ROOT/vendor/godot-forest-demo"
REPO_URL="${FOREST_DEMO_URL:-https://github.com/GamesNotDeveloped/godot-forest-demo.git}"

mkdir -p "$ROOT/vendor" "$ROOT/addons"
if [[ -d "$VENDOR_DIR/.git" ]]; then
  echo "Updating existing clone at $VENDOR_DIR"
  git -C "$VENDOR_DIR" fetch --depth 1 origin
  git -C "$VENDOR_DIR" reset --hard FETCH_HEAD
else
  echo "Shallow-cloning $REPO_URL -> $VENDOR_DIR"
  git clone --depth 1 "$REPO_URL" "$VENDOR_DIR"
fi

# Symlink path is relative to the link's directory.
link_rel() {
  local dest="$1" rel_src="$2"
  ln -sfn "$rel_src" "$dest"
  echo "  link $dest -> $rel_src"
}

cd "$ROOT"
link_rel scenery vendor/godot-forest-demo/scenery
link_rel textures vendor/godot-forest-demo/textures
link_rel materials vendor/godot-forest-demo/materials
link_rel grass vendor/godot-forest-demo/grass
link_rel levels vendor/godot-forest-demo/levels
link_rel objects vendor/godot-forest-demo/objects
link_rel addons/gnd_biomes ../vendor/godot-forest-demo/addons/gnd_biomes

echo "Forest vendor ready."
echo "  Bypass scene: res://scenes/rough_forest_play.tscn"
echo "  Flat default:  res://main.tscn (unchanged)"
