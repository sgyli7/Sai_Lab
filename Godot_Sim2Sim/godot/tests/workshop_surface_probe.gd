extends SceneTree

func _initialize() -> void:
	call_deferred("run")

func run() -> void:
	var w=load("res://atelier/workshop.gd").new()
	w.collisions_enabled=false
	root.add_child(w)
	for key in w.PALETTE:
		w.materials[key]=StandardMaterial3D.new()
		var st=SurfaceTool.new();st.begin(Mesh.PRIMITIVE_TRIANGLES);w.builders[key]=st
	w._build_solids()
	load("res://atelier/workshop_details.gd").new().build(w)
	var faces=[]
	for color in w.builders:
		var mesh=w.builders[color].commit()
		var arrays=mesh.surface_get_arrays(0)
		var vertices=arrays[Mesh.ARRAY_VERTEX]
		var indices=arrays[Mesh.ARRAY_INDEX]
		if indices==null or indices.is_empty():indices=range(vertices.size())
		for i in range(0,indices.size(),3):
			var a:Vector3=vertices[indices[i]];var b:Vector3=vertices[indices[i+1]];var c:Vector3=vertices[indices[i+2]]
			var n:Vector3=(b-a).cross(c-a).normalized()
			var axis=n.abs().max_axis_index()
			if absf(n[axis])<.999999:continue
			if (b-a).cross(c-a).length()<.000001:continue
			# Godot clockwise winding: these downward faces are buried in the floor.
			if axis==1 and n.y>0 and a.y<=.000001:continue
			faces.append({"color":color,"axis":axis,"sign":signf(n[axis]),"plane":a[axis],"vertices":[[a.x,a.y,a.z],[b.x,b.y,b.z],[c.x,c.y,c.z]]})
	FileAccess.open(OS.get_environment("MD_AUDIT_OUTPUT"),FileAccess.WRITE).store_string(JSON.stringify(faces))
	quit()
