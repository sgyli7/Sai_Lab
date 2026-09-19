extends Node3D
## Small, static contact specimens. Render triangles and convex hull use the
## same mesh and transform. No robot, solver or control parameter is altered.
var specimens: Array[Dictionary] = []
var w: Node3D

func build(workshop: Node3D, visible_geometry: bool) -> void:
	w=workshop
	# South apron, clear of spawn, the central parking bay and the east exit.
	# Approach from z < 1.28; the stair lane rises toward +Z.
	for i in range(3):
		var height:=.005*(i+1)
		var size:=Vector3(.36,height,.18)
		var p:=Vector3(.02,height*.5,1.38+i*.18)
		_add_specimen("Step_%02dmm" % int(height*1000),w._rounded_box(size,.001),p,"metal",visible_geometry)
		if visible_geometry:
			w._label("%02d mm" % int(height*1000),p+Vector3(.14,height*.5+.001,0),20,.00035,"ink",Vector3(-90,0,0))
	# One closed convex wedge, 30 mm rise over 450 mm (3.8 degrees).
	_add_specimen("Ramp_30mm",_wedge(.32,.45,.03),Vector3(.52,0,1.55),"paper",visible_geometry)
	for i in range(2):
		var height:=.01+.01*i
		_add_specimen("Threshold_%02dmm" % int(height*1000),w._rounded_box(Vector3(.28,height,.055),.002),Vector3(1.02,height*.5,1.35+i*.31),"purple",visible_geometry)
	_add_specimen("Stop_40mm",w._rounded_box(Vector3(.32,.04,.085),.004),Vector3(1.48,.02,1.60),"graphite",visible_geometry)
	if visible_geometry:
		w._label("05 / CONTACT LAB",Vector3(.67,.003,1.15),36,.00055,"graphite",Vector3(-90,0,0))
		w._label("STEPS    /    RAMP    /    THRESHOLDS",Vector3(.67,.003,1.98),24,.00045,"shadow",Vector3(-90,0,0))
		for x in [-.23,1.72]:
			w._box(Vector3(x,.002,1.58),Vector3(.012,.002,.68),"yellow",.0003)

func _add_specimen(id: String,mesh: Mesh,p: Vector3,color: String,visible_geometry: bool) -> void:
	var body:=StaticBody3D.new();body.name=id;body.position=p
	body.collision_layer=1;body.collision_mask=1
	body.physics_material_override=w.server.get_node("World/Floor").physics_material_override
	var shape:=mesh.create_convex_shape(true,false);shape.margin=.0002
	var collider:=CollisionShape3D.new();collider.shape=shape;body.add_child(collider)
	if visible_geometry:
		var visual:=MeshInstance3D.new();visual.mesh=mesh
		visual.material_override=w.materials[color];body.add_child(visual)
	add_child(body)
	specimens.append({"name":id,"body":body,"mesh":mesh})
	w.camera_obstacles.append(Transform3D(Basis.IDENTITY,p)*mesh.get_aabb())

func _wedge(width: float,length: float,height: float) -> ArrayMesh:
	var vertices: Array[Vector3]=[
		Vector3(-width*.5,0,-length*.5),Vector3(width*.5,0,-length*.5),
		Vector3(-width*.5,0,length*.5),Vector3(width*.5,0,length*.5),
		Vector3(-width*.5,height,length*.5),Vector3(width*.5,height,length*.5)]
	# Clockwise outside winding, including a closed underside.
	var triangles: Array[int]=[0,1,4,1,5,4,0,2,1,1,2,3,2,4,3,3,4,5,0,4,2,1,3,5]
	var st:=SurfaceTool.new();st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in range(0,triangles.size(),3):
		var a:=vertices[triangles[i]];var b:=vertices[triangles[i+1]];var c:=vertices[triangles[i+2]]
		var normal:Vector3=-(b-a).cross(c-a).normalized()
		for v in [a,b,c]:st.set_normal(normal);st.add_vertex(v)
	st.index();return st.commit()
