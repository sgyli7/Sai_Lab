extends "res://atelier/atelier_server.gd"

# Test-only RPC; this script is never loaded by a play or training scene.
var locked_camera: Dictionary = {}

func _follow_camera(dt: float) -> void:
	if not locked_camera.is_empty():
		var cam:=get_node("World/Camera3D") as Camera3D
		var p:Array=locked_camera.position;var b:Array=locked_camera.basis
		cam.global_transform=Transform3D(Basis(Vector3(b[0][0],b[0][1],b[0][2]),Vector3(b[1][0],b[1][1],b[1][2]),Vector3(b[2][0],b[2][1],b[2][2])),Vector3(p[0],p[1],p[2]))
		cam.fov=locked_camera.fov;cam.near=.015;cam.far=100
		return
	super._follow_camera(dt)

func _handle(cmd: Variant) -> void:
	if cmd.get("cmd","")=="atelier_camera_lock":
		locked_camera=cmd.camera;_follow_camera(0.)
		_send_dict({"ok":true});return
	if cmd.get("cmd","")=="screenshot":
		_follow_camera(0.);RenderingServer.force_draw(true)

	if cmd.get("cmd","")!="shadow_probe":
		super._handle(cmd);return
	_freeze(true) # Hold the settled pose across render-only parameter changes.
	var sun:=get_node("World/Sun") as DirectionalLight3D
	for key in ["shadow_bias","shadow_normal_bias","shadow_enabled","directional_shadow_max_distance","directional_shadow_mode"]:
		if cmd.has(key):sun.set(key,cmd[key])
	if cmd.get("fixture",false) and not has_node("ContactFixture"):
		var box:=MeshInstance3D.new();box.name="ContactFixture"
		var mesh:=SphereMesh.new();mesh.radius=.015;mesh.height=.03;box.mesh=mesh
		box.position=Vector3(.25,.015,0);add_child(box)
	var cam:=get_node("World/Camera3D") as Camera3D
	var meshes:Array=[]
	for mesh in get_node("RobotHost").find_children("vis_*","MeshInstance3D",true,false):
		var min_y:=INF
		var point:=Vector3.ZERO
		for surface in range(mesh.mesh.get_surface_count()):
			var vertices:PackedVector3Array=mesh.mesh.surface_get_arrays(surface)[Mesh.ARRAY_VERTEX]
			for v in vertices:
				var p:Vector3=mesh.global_transform*v
				if p.y<min_y:min_y=p.y;point=p
		var screen:=cam.unproject_position(point)
		if min_y<.03:meshes.append({"path":str(mesh.get_path()),"min_y":min_y,"point":[point.x,point.y,point.z],"pixel":[screen.x,screen.y]})
	var points:Array=[]
	# Samples outside the flush box, along the projected light direction.
	var ray:=-sun.global_basis.z;ray.y=0;ray=ray.normalized()
	var edge:=0.0
	for mm in [1,2,3,5,10,20,30,40,50,70]:
		var p:Vector3=Vector3(.25,0,0)+ray*(edge+float(mm)*.001)
		var screen:=cam.unproject_position(p)
		points.append({"mm":mm,"pixel":[screen.x,screen.y]})
	_send_dict({"ok":true,"feet":meshes,"samples":points,"sun_basis_z":[sun.global_basis.z.x,sun.global_basis.z.y,sun.global_basis.z.z],"bias":sun.shadow_bias,"normal_bias":sun.shadow_normal_bias,"distance":sun.directional_shadow_max_distance,"mode":sun.directional_shadow_mode})
