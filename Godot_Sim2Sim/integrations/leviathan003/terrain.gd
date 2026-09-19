extends "res://polar_range/terrain.gd"
## Same game terrain grid, surface and collision; no render-buffer readback.
## Godot 4.7.2/GB10 intermittently crashes in newly committed Mesh.get_faces().
## Source terrain SHA256: 58d22fdd63a72e2cefe029388474dc73992093db602fc9bd6f7e8b3c5bf49faa
## Only build is specialized. Height sampling and grid lookup remain inherited.
func build(scene:Node3D) -> void:
	xs=_coordinates(134.);zs=_coordinates(0.)
	# Exact apron boundaries prevent a raised lip across its planar interior.
	for x in [78.,190.]:
		if not xs.has(x):xs.append(x)
	for z in [-30.,30.]:
		if not zs.has(z):zs.append(z)
	xs.sort();zs.sort()
	for x in xs:
		for z in zs:heights.append(height_at(x,z))
	var collision_faces:=PackedVector3Array()
	collision_faces.resize((xs.size()-1)*(zs.size()-1)*6)
	var face_index:=0
	var st:=SurfaceTool.new();st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in range(xs.size()-1):
		for j in range(zs.size()-1):
			var a:=_vertex(i,j);var b:=_vertex(i+1,j);var c:=_vertex(i,j+1);var d:=_vertex(i+1,j+1)
			for v:Vector3 in [a,b,c,b,d,c]:
				st.add_vertex(v)
				# Match TriangleMesh::create rounding used by Mesh.get_faces().
				collision_faces[face_index]=v.snappedf(.0001);face_index+=1
	triangle_count=(xs.size()-1)*(zs.size()-1)*2
	st.generate_normals();st.index();var mesh:=st.commit()
	var body:=StaticBody3D.new();body.name="PolarSnow";body.collision_layer=11;body.collision_mask=5
	body.add_to_group("sai_driving_surface")
	body.physics_material_override=scene.server.get_node("World/Floor").physics_material_override
	var shape:=ConcavePolygonShape3D.new();shape.set_faces(collision_faces);shape.backface_collision=true
	var collider:=CollisionShape3D.new();collider.shape=shape;body.add_child(collider);scene.add_child(body)
	if scene.visuals_enabled:
		var visual:=MeshInstance3D.new();visual.name="ContinuousSnow";visual.mesh=mesh
		var material:=ShaderMaterial.new();material.shader=load("res://polar_range/snow.gdshader")
		visual.material_override=material;visual.cast_shadow=GeometryInstance3D.SHADOW_CASTING_SETTING_OFF;body.add_child(visual)
