extends Node3D
## Force TerrainPatch3D mesh/collision generation, then shift the patch so the
## surface at local (0,0) sits at world y=0. Runs as a World sibling during
## children _ready — before physics_server.gd instances the robot — so Python
## reset_z (~0.125) lands on terrain instead of a noise bump/dip.

@export var terrain_path: NodePath = NodePath("../ForestTerrain")

func _ready() -> void:
	_align_sync()


func _align_sync() -> void:
	var t := get_node_or_null(terrain_path) as Node3D
	if t == null:
		push_warning("terrain_spawn_align: missing terrain at %s" % str(terrain_path))
		return
	# TerrainPatch3D normally defers regenerate; force sync now.
	if t.has_method("_generate"):
		t.call("_generate")
	elif t.has_method("_deferred_regenerate"):
		t.call("_deferred_regenerate")
	if not t.has_method("_sample_height"):
		push_warning("terrain_spawn_align: terrain has no _sample_height")
		return
	var y0: float = float(t.call("_sample_height", 0.0, 0.0))
	t.position.y = -y0
	print("rough_forest: aligned ForestTerrain so origin surface y=0 (raw y0=%.4f)" % y0)
