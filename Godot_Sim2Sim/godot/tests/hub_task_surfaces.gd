extends SceneTree
## Exercise the actual task builder without a robot, controller or renderer.
class TaskHub extends "res://hub/hub.gd":
	func _ready() -> void:pass
	func _process(_dt:float) -> void:pass
	func _physics_process(_dt:float) -> void:pass
class Materials extends Node3D:
	var materials:Dictionary={}

func _initialize() -> void:call_deferred("run")
func run() -> void:
	var hub:=TaskHub.new();root.add_child(hub)
	var palette:=Materials.new();hub.add_child(palette);hub.atelier=palette
	for key in ["purple","paper","floor"]:palette.materials[key]=StandardMaterial3D.new()
	hub._headless=false
	var report:Dictionary={};var errors:Array=[]
	for task in hub.TASKS:
		hub.active_task=task;hub._build_task_course()
		var coplanar:=0;var area:=0.;var contract:Array=[];var triangles:=0;var visible_boxes:=0
		for zone in hub.course_root.get_children():
			for body in zone.get_children():
				if not body is StaticBody3D:continue
				var shape:CollisionShape3D=body.get_child(0)
				contract.append({"name":str(zone.name)+"/"+str(body.name),"p":str(body.global_position),"size":str(shape.shape.size),"margin":shape.shape.margin,"layer":body.collision_layer,"mask":body.collision_mask})
				for child in body.get_children():
					if not child is MeshInstance3D:continue
					visible_boxes+=1
					var bounds:AABB=child.global_transform*child.mesh.get_aabb()
					var expected_top:float=body.global_position.y+shape.shape.size.y*.5
					if absf(bounds.end.y-expected_top)>.000001:errors.append(task+": visible step height changed")
					if bounds.position.y<-.000001:errors.append(task+": underground render geometry remains")
					if absf(bounds.size.x-shape.shape.size.x)>.000001 or absf(bounds.size.z-shape.shape.size.z)>.000001:errors.append(task+": task footprint changed")
					var faces:PackedVector3Array=child.mesh.get_faces()
					for i in range(0,faces.size(),3):
						var a:Vector3=child.global_transform*faces[i]
						var b:Vector3=child.global_transform*faces[i+1]
						var c:Vector3=child.global_transform*faces[i+2]
						triangles+=1
						if maxf(absf(a.y),maxf(absf(b.y),absf(c.y)))<.000001:
							coplanar+=1;area+=(b-a).cross(c-a).length()*.5
		if visible_boxes!=18 or triangles!=180:errors.append(task+": exposed task surfaces missing or duplicated")
		if contract.size()!=21:errors.append(task+": task colliders missing")
		if coplanar>0:errors.append(task+": %d triangles overlap the floor (%.4f m²)"%[coplanar,area])
		report[task]={"coplanar_triangles":coplanar,"overlap_area":area,"triangles":triangles,"physics":contract}
	var target:=OS.get_environment("HUB_SURFACE_REPORT")
	if target!="":FileAccess.open(target,FileAccess.WRITE).store_string(JSON.stringify(report,"  "))
	print("TASK_SURFACE_AUDIT ",JSON.stringify(errors));quit(0 if errors.is_empty() else 1)
