extends Node
## Workshop-only release mesh paint. No control or physics changes.
const COLORS := {
	"graphite": Color("6c6a68"), "yellow": Color("e0bd38"),
	"purple": Color("89729e"), "rubber": Color("39383d"),
	"metal": Color("969b95"), "paper": Color("c6c4b7"),
	"blue": Color("517fa4"), "floor": Color("b2b1a2")
}
var scene: Node3D
var pigments: Dictionary = {}

func _make_materials() -> void:
	for key in COLORS:
		var material := ShaderMaterial.new()
		material.shader = load("res://visuals/microduck/enamel.gdshader")
		material.set_shader_parameter("pigment", COLORS[key])
		material.set_shader_parameter("hatch_strength", .03)
		var ink := ShaderMaterial.new()
		ink.shader = load("res://visuals/microduck/ink.gdshader")
		ink.set_shader_parameter("line_pixels", .60)
		material.next_pass = ink
		pigments[key] = material

func _paint_robot() -> void:
	for name in scene.robot.bodies:
		var body: RigidBody3D = scene.robot.bodies[name]
		var descriptions: Array = scene.specification.bodies[name].visuals
		for mesh in body.find_children("*", "MeshInstance3D", true, false):
			var index := int(str(mesh.name).get_slice("_", str(mesh.name).get_slice_count("_")-1))
			var rgba: Array = descriptions[index].rgba
			var key := "graphite"
			if rgba[2] > .5 and rgba[0] < .1:
				key = "blue" # Preserve the release's identifiable open cargo bay.
			elif rgba[0] < .045:
				key = "rubber"
			elif str(name).begins_with("arm_") and rgba[0] > .6:
				key = "yellow"
			elif rgba[0] > .5:
				key = "metal"
			elif rgba[1] > rgba[0] * 1.5:
				key = "purple"
			elif str(name).ends_with("_wheel") and index == 2:
				key = "yellow"
			mesh.material_override = pigments[key]
	if scene.robot.item != null:
		for mesh in scene.robot.item.find_children("*", "MeshInstance3D", true, false):
			mesh.material_override = pigments.yellow
