extends RefCounted
## Map 03: continuous hard snow. One shared height grid for visuals and Jolt.
var xs:Array[float]=[]
var zs:Array[float]=[]
var heights:PackedFloat64Array=[]
var triangle_count:=0

static func height_at(x:float,z:float) -> float:
	var entry:=smoothstep(192.,210.,x)
	var first:=.8*pow(sin(clampf((x-200.)/110.,0.,1.)*PI),2.)
	var rolling:=3.5*pow(sin((x-280.)/155.),2.)*smoothstep(280.,360.,x)
	var crossfall:=1.8*sin(z/130.)*sin((x-200.)/190.)
	var snow:=3.+entry*(first+rolling+crossfall)
	snow+=1.2*sin(x/65.)*sin(z/80.)*smoothstep(55.,130.,absf(z))
	var distance:=Vector2(x-134.,z).length()
	snow+=smoothstep(160.,600.,distance)*2.5*sin(x/210.)*sin(z/280.)
	# A few isolated distant peaks to the north-west; the other horizons stay open.
	for peak in [Vector3(-6200.,420.,-9800.),Vector3(-8300.,610.,-11300.),Vector3(-10300.,370.,-10500.)]:
		var p:=Vector2((x-peak.x)/1700.,(z-peak.z)/1300.)
		snow+=peak.y*exp(-p.length_squared()*1.7)*(1.+.12*sin(x*.006+z*.004))
	# Flat full-size departure area; the very gentle shoulder is part of the mesh.
	var q:Vector2=(Vector2(x,z)-Vector2(134,0)).abs()-Vector2(56,30)
	var pad_distance:float=q.max(Vector2.ZERO).length()+minf(maxf(q.x,q.y),0.)
	return lerpf(3.,snow,smoothstep(0.,24.,pad_distance))

func _coordinates(center:float) -> Array[float]:
	var positive:Array[float]=[0.]
	var d:=0.
	while d<40000.:
		var step:float=4. if d<160. else 8. if d<800. else 40. if d<4000. else 200. if d<16000. else 1000.
		d=minf(40000.,d+step);positive.append(d)
	var values:Array[float]=[]
	for i in range(positive.size()-1,0,-1):values.append(center-positive[i])
	for v in positive:values.append(center+v)
	return values

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
	var st:=SurfaceTool.new();st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in range(xs.size()-1):
		for j in range(zs.size()-1):
			var a:=_vertex(i,j);var b:=_vertex(i+1,j);var c:=_vertex(i,j+1);var d:=_vertex(i+1,j+1)
			for v in [a,b,c,b,d,c]:st.add_vertex(v)
	triangle_count=(xs.size()-1)*(zs.size()-1)*2
	st.generate_normals();st.index();var mesh:=st.commit()
	var body:=StaticBody3D.new();body.name="PolarSnow";body.collision_layer=11;body.collision_mask=5
	body.add_to_group("sai_driving_surface")
	body.physics_material_override=scene.server.get_node("World/Floor").physics_material_override
	var shape:=ConcavePolygonShape3D.new();shape.set_faces(mesh.get_faces());shape.backface_collision=true
	var collider:=CollisionShape3D.new();collider.shape=shape;body.add_child(collider);scene.add_child(body)
	if scene.visuals_enabled:
		var visual:=MeshInstance3D.new();visual.name="ContinuousSnow";visual.mesh=mesh
		var material:=ShaderMaterial.new();material.shader=load("res://polar_range/snow.gdshader")
		visual.material_override=material;visual.cast_shadow=GeometryInstance3D.SHADOW_CASTING_SETTING_OFF;body.add_child(visual)

func _vertex(i:int,j:int) -> Vector3:
	return Vector3(xs[i],heights[i*zs.size()+j],zs[j])

func ground(x:float,z:float) -> float:
	if xs.is_empty():return height_at(x,z)
	var i:=clampi(xs.bsearch(x)-1,0,xs.size()-2)
	var j:=clampi(zs.bsearch(z)-1,0,zs.size()-2)
	var u:=clampf((x-xs[i])/(xs[i+1]-xs[i]),0.,1.)
	var v:=clampf((z-zs[j])/(zs[j+1]-zs[j]),0.,1.)
	var a:float=heights[i*zs.size()+j];var b:float=heights[(i+1)*zs.size()+j]
	var c:float=heights[i*zs.size()+j+1];var d:float=heights[(i+1)*zs.size()+j+1]
	return a*(1.-u-v)+b*u+c*v if u+v<=1. else b*(1.-v)+c*(1.-u)+d*(u+v-1.)
