extends RefCounted
## Original 3D geology and a distant field observatory. No backdrop cards.
var w:Node3D
var terrain_faces:Array[Vector3]=[]
var ground_cells:Dictionary={}
const OBSERVATORY_SITE:=Vector3(-35.,1.1,-20.4)
var observatory_position:=OBSERVATORY_SITE

func build(station:Node3D) -> void:
	w=station
	var surface:=ShaderMaterial.new();surface.shader=load("res://science_station/ground.gdshader")
	surface.set_shader_parameter("station_influence",0.)
	var rings:Array[float]=[1.,1.20,1.52,1.95,2.55,3.5,5.5,9.,18.]
	var st:=SurfaceTool.new();st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for band in range(rings.size()-1):
		for i in range(160):
			var a:float=i*TAU/160.;var b:float=(i+1)*TAU/160.
			var p:=ridge(a,rings[band],band);var q:=ridge(b,rings[band],band)
			var r:=ridge(a,rings[band+1],band+1);var s:=ridge(b,rings[band+1],band+1)
			_ground_triangle(st,p,r,q)
			_ground_triangle(st,q,r,s)
	_index_ground()
	st.generate_normals();st.index()
	if w.visuals_enabled:
		var terrain:=MeshInstance3D.new();terrain.name="DistantTerrain"
		terrain.mesh=st.commit();terrain.material_override=surface;w.add_child(terrain)
	var body:=StaticBody3D.new();body.name="DistantGround"
	body.collision_layer=3;body.collision_mask=5;body.add_to_group("sai_driving_surface")
	body.physics_material_override=w.server.get_node("World/Floor").physics_material_override
	var shape:=ConcavePolygonShape3D.new();shape.set_faces(PackedVector3Array(terrain_faces));shape.backface_collision=true
	var collider:=CollisionShape3D.new();collider.shape=shape;body.add_child(collider);w.add_child(body)
	# Close the remote rim below sight lines from elevated overview cameras.
	var skirt:=SurfaceTool.new();skirt.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in range(160):
		var a:=ridge(i*TAU/160.,18.,8);var b:=ridge((i+1)*TAU/160.,18.,8)
		for v in [a,a-Vector3(0,200,0),b,b,a-Vector3(0,200,0),b-Vector3(0,200,0)]:skirt.add_vertex(v)
	skirt.generate_normals();w._add(skirt.commit(),Vector3.ZERO,"distant")
	# Low eroded ledges break the boundary into rock shelves and wind corridors.
	for i in range(38):
		var a:float=i*2.399
		var ring:float=1.02+.027*(i%6)
		var p:=ridge(a,ring,0)
		var size:=Vector3(1.1+(i%5)*.45,.25+(i%4)*.26,.55+(i%3)*.3)
		p.y=foundation(p,size)-.12
		outcrop(w,p,size,i*.71)
	# Broad, broken escarpments sit behind the towers, with quiet gaps for sky.
	for spec in [[Vector3(9,0,-49),Vector3(11,6.4,5.8),.5],
		[Vector3(-15,0,-56),Vector3(9,7.8,6.),1.8],
		[Vector3(55,0,2),Vector3(5.8,6.,11),2.9],
		[Vector3(-9,0,41),Vector3(13,4.2,6),4.]]:
		var p:Vector3=spec[0];p.y=foundation(p,spec[1])-.25
		outcrop(w,p,spec[1],spec[2],true)
	# Three articulated silhouettes; gaps between them preserve the open horizon.
	for spec in [[Vector3(-31,1.2,4),Vector3(3.2,3.2,2.6),1.7],
		[Vector3(35,1.3,-29),Vector3(5.3,4.7,3.2),3.1],
		[Vector3(27,1.1,19),Vector3(3.9,3.1,2.7),2.2]]:
		var p:Vector3=spec[0];p.y=foundation(p,spec[1])-.20
		outcrop(w,p,spec[1],spec[2])
	var previous:bool=w.solid_scope;w.solid_scope=true
	_observatory(OBSERVATORY_SITE)
	# A quiet horizon relay carries the same enclosure/antenna vocabulary.
	var p:=Vector3(40,0,-39);p.y=foundation(p,Vector3(2,0,2))
	w._box(p+Vector3(0,.9,0),Vector3(2.8,1.8,2.2),"paper",.16)
	w._box(p+Vector3(0,1.88,0),Vector3(3.2,.12,2.5),"blue",.04)
	w._cylinder(p+Vector3(-.7,2.2,0),.45,.6,"metal")
	w._dome(p+Vector3(-.7,2.5,0),.6,.30,"paper")
	for x in [-1.,1.]:w._line(p+Vector3(x,1.8,0),p+Vector3(x,4.+x*.5,0),.015,"graphite")
	w.solid_scope=previous

func _prepared_vertex(v:Vector3) -> Vector3:
	# The weather station occupies a graded patch of continuous terrain.
	var q:Vector2=Vector2(v.x-OBSERVATORY_SITE.x,v.z-OBSERVATORY_SITE.z).abs()-Vector2(3.7,2.6)
	var distance:float=q.max(Vector2.ZERO).length()+minf(maxf(q.x,q.y),0.)
	v.y=lerpf(OBSERVATORY_SITE.y,v.y,smoothstep(.05,2.3,distance))
	return v

func _ground_triangle(st:SurfaceTool,a:Vector3,b:Vector3,c:Vector3) -> void:
	var bounds:=Rect2(Vector2(a.x,a.z),Vector2.ZERO).expand(Vector2(b.x,b.z)).expand(Vector2(c.x,c.z))
	var near_site:bool=bounds.intersects(Rect2(Vector2(OBSERVATORY_SITE.x-6.2,OBSERVATORY_SITE.z-5.1),Vector2(12.4,10.2)))
	var cuts:int=16 if near_site else 1
	for i in range(cuts):
		for j in range(cuts-i):
			var u:Vector3=a+(b-a)*float(i)/cuts+(c-a)*float(j)/cuts
			var v:Vector3=a+(b-a)*float(i+1)/cuts+(c-a)*float(j)/cuts
			var t:Vector3=a+(b-a)*float(i)/cuts+(c-a)*float(j+1)/cuts
			for point in [u,v,t]:
				point=_prepared_vertex(point);st.add_vertex(point);terrain_faces.append(point)
			if i+j<cuts-1:
				var r:Vector3=a+(b-a)*float(i+1)/cuts+(c-a)*float(j+1)/cuts
				for point in [v,r,t]:
					point=_prepared_vertex(point);st.add_vertex(point);terrain_faces.append(point)

func ridge(a:float,r:float,band:int) -> Vector3:
	var dx:float=cos(a);var dz:float=sin(a)
	var square:float=1./maxf(absf(dx),absf(dz))
	var wave:float=.10*sin(a*5.+band*.72)+.05*cos(a*11.+band*.39)+.025*sin(a*23.-band)
	var warp:float=1.+wave*minf(1.,(r-1.)*2.)
	# Integer angular frequency closes the 0/TAU seam in both height and slope.
	var peak:float=pow(maxf(0.,sin(a*3.+band*.61)),2.)
	var ridge_h:float=[0.,.7,2.5,4.8,7.3,10.8,13.,16.,18.][band]
	var y:float=ridge_h*(.55+peak*.65+wave*2.)
	return Vector3(dx*square*24.*r*warp,y,dz*square*20.*r*warp-6.)

func _index_ground() -> void:
	# Nearby camera collision queries sample many points per frame. Keep them
	# local even though the graded outpost uses a denser piece of the mesh.
	for i in range(0,terrain_faces.size(),3):
		var a:Vector3=terrain_faces[i];var b:Vector3=terrain_faces[i+1];var c:Vector3=terrain_faces[i+2]
		var low:Vector3=a.min(b).min(c);var high:Vector3=a.max(b).max(c)
		for x in range(maxi(-10,floori(low.x/8.)),mini(10,floori(high.x/8.))+1):
			for z in range(maxi(-10,floori(low.z/8.)),mini(10,floori(high.z/8.))+1):
				var key:=Vector2i(x,z)
				if not ground_cells.has(key):ground_cells[key]=[]
				ground_cells[key].append(i)

func ground(x:float,z:float) -> float:
	var p:=Vector2(x,z)
	var key:=Vector2i(floori(x/8.),floori(z/8.))
	var candidates:Array=ground_cells.get(key,[])
	if not ground_cells.has(key):candidates=range(0,terrain_faces.size(),3)
	for i in candidates:
		var a:=Vector2(terrain_faces[i].x,terrain_faces[i].z)
		var b:=Vector2(terrain_faces[i+1].x,terrain_faces[i+1].z)
		var c:=Vector2(terrain_faces[i+2].x,terrain_faces[i+2].z)
		var det:float=(b-a).cross(c-a)
		if absf(det)<.0001:continue
		var u:float=(p-a).cross(c-a)/det;var v:float=(b-a).cross(p-a)/det
		if u>=-.00001 and v>=-.00001 and u+v<=1.00001:
			return terrain_faces[i].y*(1.-u-v)+terrain_faces[i+1].y*u+terrain_faces[i+2].y*v
	return 0.

func foundation(p:Vector3,size:Vector3) -> float:
	var y:float=ground(p.x,p.z)
	for i in range(13):
		var a:float=i*TAU/13.
		y=minf(y,ground(p.x+cos(a)*size.x,p.z+sin(a)*size.z))
	return y

static func outcrop(station:Node3D,p:Vector3,size:Vector3,seed:float,mesa:bool=false) -> ArrayMesh:
	var count:=13
	var profile:Array=[1.02,1.04,.94,.84,.86,.83] if mesa else [1.02,1.05,.89,.92,.65,.33]
	var levels:Array[PackedVector3Array]=[]
	for j in range(profile.size()):
		var ring:=PackedVector3Array()
		for i in range(count):
			var a:float=i*TAU/count
			var wobble:float=1.+.12*sin(a*3.+seed)+.07*cos(a*5.-seed*.8)+.025*sin(a*7.+j*1.1)
			var y:float=size.y*(j/5.+.055*sin(a*3.+seed+j*.7)+.024*cos(a*7.-seed))
			ring.append(p+Vector3(cos(a)*size.x*wobble*profile[j]+j*.045*size.x,y,sin(a)*size.z*wobble*profile[j]))
		levels.append(ring)
	var hull:=SurfaceTool.new();hull.begin(Mesh.PRIMITIVE_TRIANGLES)
	for j in range(levels.size()-1):
		var st:=SurfaceTool.new();st.begin(Mesh.PRIMITIVE_TRIANGLES)
		for i in range(count):
			var n:int=(i+1)%count
			for v in [levels[j][i],levels[j][n],levels[j+1][i],levels[j][n],levels[j+1][n],levels[j+1][i]]:
				st.add_vertex(v);hull.add_vertex(v)
		st.generate_normals();st.index();station._add(st.commit(),Vector3.ZERO,(["silt","rock","rock","rock","chalk"] if mesa else ["silt","rock","chalk","rock","chalk"])[j])
		if j in [1,3]:
			for i in range(count):
				if (i+int(seed*3.))%4==0:continue
				var start:Vector3=levels[j+1][i]
				var finish:Vector3=levels[j+1][(i+1)%count]
				station._line(start+Vector3(0,.008,0),start.lerp(finish,.79)+Vector3(0,.008,0),.006,"strata")
	var cap:=SurfaceTool.new();cap.begin(Mesh.PRIMITIVE_TRIANGLES)
	var center:=Vector3.ZERO
	for v in levels[-1]:center+=v/count
	for i in range(count):
		for v in [center,levels[-1][i],levels[-1][(i+1)%count]]:cap.add_vertex(v);hull.add_vertex(v)
	cap.generate_normals();cap.index();station._add(cap.commit(),Vector3.ZERO,"chalk")
	hull.generate_normals();hull.index();var rock_mesh:=hull.commit()
	# With the old rock fence removed, these outcrops can be approached. Use
	# their displayed faces rather than invisible convex caps across ledges.
	if station.collisions_enabled and station.collision_body!=null:
		var shape:=ConcavePolygonShape3D.new();shape.set_faces(rock_mesh.get_faces());shape.backface_collision=true
		var collider:=CollisionShape3D.new();collider.name="Outcrop_%d"%station.collision_body.get_child_count();collider.shape=shape
		# Keep scenery in the existing static compound, as the station buildings
		# are. Extra stationary bodies must not change robot/prop body allocation.
		station.collision_body.add_child(collider)
	station.camera_obstacles.append(rock_mesh.get_aabb())
	# Broken ink strokes follow individual triangulated rock faces. Broad faces
	# stay quiet; small fractures cluster around the projecting ledges.
	var width:float=clampf(size.y*.002,.002,.011)
	for i in range(count):
		if i%3==2:continue
		var n:int=(i+1)%count
		var outward:=Vector3(cos((i+.4)*TAU/count),0,sin((i+.4)*TAU/count))*.012
		for j in range(1,5):
			var a:Vector3=levels[j][i].lerp(levels[j][n],.32)
			var b:Vector3=levels[j+1][i].lerp(levels[j+1][n],.32)
			# Split at the triangle diagonal so strokes do not cut through a face.
			var mid:Vector3=levels[j][n].lerp(levels[j+1][i],.68)
			station._line(a.lerp(mid,.12)+outward,mid+outward,width,"strata")
			station._line(mid+outward,mid.lerp(b,.78)+outward,width,"strata")
			if (i+j)%3==0:
				station._line(mid+outward,mid.lerp(levels[j+1][n],.30)+outward,width*.7,"strata")

	return rock_mesh

func _observatory(p:Vector3) -> void:
	load("res://science_station/ridge_observatory.gd").new().build(w,p)
