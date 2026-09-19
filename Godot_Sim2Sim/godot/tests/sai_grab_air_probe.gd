extends SceneTree
## A selected object at the tool point must not attach without gripper contact.
class FakeRobot extends Node3D:
	var bodies: Dictionary = {}
	func gv(v: Array) -> Vector3:
		return Vector3(float(v[0]), float(v[1]), float(v[2]))
	func sim_time_seconds() -> float:
		return 0.0
class FakeProps extends Node3D:
	var items: Array = []
class FakeAtelier extends Node3D:
	var loose_props: FakeProps
class FakeHub extends Node3D:
	var _headless := true
	var _t := 0.0
	var atelier: FakeAtelier
	func _stage_label(stage: String) -> String:
		return stage
class FakeScene extends Node3D:
	var robot: FakeRobot
	var hub: FakeHub
	var specification := {"tool_local_m": [0.0, 0.0, 0.0]}

func _initialize() -> void:
	call_deferred("run")

func run() -> void:
	var scene := FakeScene.new()
	root.add_child(scene)
	var hub := FakeHub.new()
	scene.add_child(hub)
	scene.hub = hub
	var atelier := FakeAtelier.new()
	hub.add_child(atelier)
	hub.atelier = atelier
	var props := FakeProps.new()
	atelier.add_child(props)
	atelier.loose_props = props
	var robot := FakeRobot.new()
	scene.add_child(robot)
	scene.robot = robot
	for name in ["chassis", "arm_gripper", "arm_moving_jaw"]:
		var body := RigidBody3D.new()
		body.name = name
		robot.add_child(body)
		robot.bodies[name] = body
	var item := RigidBody3D.new()
	item.name = "Bottle6g"
	item.contact_monitor = true
	item.max_contacts_reported = 8
	item.set_meta("grasp_center", Vector3.ZERO)
	item.set_meta("grasp_size", Vector3(.02, .03, .02))
	scene.add_child(item)
	props.items.append({"body": item})
	var grab: Variant = load("res://hub/scene_grab.gd").new()
	grab.scene = scene
	scene.add_child(grab)
	grab.target = item
	grab.busy = true
	await physics_frame
	grab.accept({"assist_grip": true, "grab_stage": "manipulation"})
	var attached := false
	for child in grab.get_children():
		if child is Joint3D: attached = true
	var result := {"no_contact": item.get_colliding_bodies().is_empty(),
		"attached": attached, "held": grab.held}
	print("SAI_AIR_GRAB_RESULT ", JSON.stringify(result))
	quit(1 if result.attached or result.held else 0)
