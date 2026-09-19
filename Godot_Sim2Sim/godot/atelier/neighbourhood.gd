extends RefCounted
## Complete surrounding district, in metres. Continuous visible perimeter owns its collision.
const SEGMENTS:=64
const RADIUS:=9.0
var w: Node3D

func build(workshop: Node3D) -> void:
	w=workshop
	if OS.get_environment("MD_NEIGHBOURHOOD")=="0":return
	var previous:bool=w.solid_scope
	w.solid_scope=true
	_perimeter()
	for i in range(16):_house(i)
	_street_furniture()
	w.solid_scope=false
	if w.visuals_enabled:_landscape()
	w.solid_scope=previous

func _point(a:float,r:float,y:float) -> Vector3:return Vector3(sin(a)*r,y,cos(a)*r)

func _quad(st:SurfaceTool,a:Vector3,b:Vector3,c:Vector3,d:Vector3) -> void:
	for v in [a,b,c,a,c,d]:st.add_vertex(v)

func _perimeter() -> void:
	# Extruded polygon ring, not a convex hull across the playable district.
	var st:=SurfaceTool.new();st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in range(SEGMENTS):
		var a:float=i*TAU/SEGMENTS;var b:float=(i+1)*TAU/SEGMENTS
		var p:=_point(a,RADIUS,0);var q:=_point(b,RADIUS,0)
		var po:=_point(a,RADIUS+.19,0);var qo:=_point(b,RADIUS+.19,0)
		var up:=Vector3(0,1.08,0)
		_quad(st,p,q,q+up,p+up)
		_quad(st,qo,po,po+up,qo+up)
		_quad(st,p+up,q+up,qo+up,po+up)
		# Broad exterior piers break up the silhouette without blocking inner paths.
		if i%4==0:w._box(_point(a,RADIUS+.10,.59),Vector3(.23,1.18,.25),"graphite",.012,a)
		w.camera_obstacles.append(AABB(p.min(q),p.max(q)-p.min(q)+Vector3(.001,1.08,.001)).grow(.10))
	# Workshop batches already contain indexed primitives. append_from() does
	# not synthesize indices for unindexed sources, so index our surfaces first.
	st.generate_normals();st.index();var mesh:=st.commit();w._add(mesh,Vector3.ZERO,"shadow")
	if w.collisions_enabled:
		var shape:=ConcavePolygonShape3D.new();shape.set_faces(mesh.get_faces());shape.backface_collision=true
		var collision:=CollisionShape3D.new();collision.name="DistrictPerimeter";collision.shape=shape
		w.collision_body.add_child(collision)

func _house(index:int) -> void:
	var angle:float=index*TAU/16
	var basis:=Basis(Vector3.UP,angle)
	var center:=_point(angle,7.55+.22*sin(index*3.1),0)
	var height:float=1.35+.19*(index%4)
	var width:float=2.15+.09*(index%3)
	var depth:=1.12
	var pigment:String="paper" if index%3==0 else "metal" if index%3==1 else "shadow"
	_box(center,basis,Vector3(0,height*.5,0),Vector3(width,height,depth),pigment,.028,angle)
	_box(center,basis,Vector3(0,height+.035,0),Vector3(width+.14,.07,depth+.13),"graphite",.010,angle)
	_box(center,basis,Vector3(0,.085,-.592),Vector3(width+.06,.17,.075),"graphite",.008,angle)
	# Recessed shutter, transom windows, drains, roof seams and service fittings.
	_box(center,basis,Vector3(-.37,.50,-.574),Vector3(.57,.78,.036),"graphite",.008,angle)
	_box(center,basis,Vector3(-.37,.50,-.598),Vector3(.51,.72,.018),"metal",.004,angle)
	for j in range(9):_box(center,basis,Vector3(-.37,.19+j*.074,-.611),Vector3(.48,.009,.007),"shadow",.001,angle)
	for x in [.25,.68]:
		_box(center,basis,Vector3(x,.83,-.578),Vector3(.35,.37,.030),"graphite",.006,angle)
		_box(center,basis,Vector3(x,.83,-.599),Vector3(.295,.305,.014),"light" if (index+int(x*10))%3==0 else "screen",.002,angle)
		_box(center,basis,Vector3(x,.83,-.611),Vector3(.012,.30,.012),"metal",.001,angle)
		_box(center,basis,Vector3(x,.655,-.62),Vector3(.39,.026,.095),"paper",.003,angle)
	_box(center,basis,Vector3(-.37,1.03,-.596),Vector3(.61,.12,.025),"purple" if index%4==0 else "paper",.004,angle)
	for x in [-width*.43,width*.43]:
		w._line(center+basis*Vector3(x,.12,-.62),center+basis*Vector3(x,height+.05,-.62),.019,"graphite")
		for y in [.30,.9]:w._cylinder(center+basis*Vector3(x,y,-.62),.025,.022,"metal")
	for j in range(7):_box(center,basis,Vector3(-width*.44+j*width*.145,height+.078,0),Vector3(.016,.018,depth+.09),"metal",.003,angle)
	w._cylinder(center+basis*Vector3(.44,height+.20,.15),.085,.30,"graphite")
	w._cylinder(center+basis*Vector3(.44,height+.365,.15),.128,.035,"paper")
	if index%3==0:
		w._cylinder(center+basis*Vector3(-.53,height+.21,.05),.18,.32,"metal")
		w._cylinder(center+basis*Vector3(-.53,height+.385,.05),.19,.026,"purple")
	if w.visuals_enabled:
		w._label("MD / %02d" % (index+10),center+basis*Vector3(-.37,1.03,-.615),32,.001,"graphite",Vector3(0,rad_to_deg(angle)+180,0))
	# Thin canopy suspended on visibly supported braces.
	_box(center,basis,Vector3(-.30,1.18,-.76),Vector3(1.10,.034,.44),"purple" if index%4==0 else "paper",.008,angle)
	for x in [-.78,.18]:w._line(center+basis*Vector3(x,.95,-.59),center+basis*Vector3(x,1.16,-.96),.008,"graphite")

func _box(center:Vector3,basis:Basis,p:Vector3,size:Vector3,color:String,bevel:float,angle:float) -> void:
	w._box(center+basis*p,size,color,bevel,angle)

func _street_furniture() -> void:
	# Four readable routes from the small workshop into a surrounding service lane.
	for i in range(8):
		var a:float=(i+.5)*TAU/8
		var p:=_point(a,5.30,0)
		w._cylinder(p+Vector3(0,.04,0),.105,.08,"graphite")
		w._cylinder(p+Vector3(0,.70,0),.027,1.32,"graphite")
		w._line(p+Vector3(0,1.35,0),p+Vector3(.17*cos(a),1.46,-.17*sin(a)),.024,"metal")
		var lamp:=p+Vector3(.17*cos(a),1.44,-.17*sin(a))
		w._cylinder(lamp,.12,.045,"graphite")
		w._cylinder(lamp-Vector3(0,.036,0),.088,.026,"light")
		# Low service cabinets are outside the central yard and real obstacles.
		var q:=_point(a+.09,5.9,0)
		w._box(q+Vector3(0,.24,0),Vector3(.38,.48,.28),"shadow",.018,a)
		w._box(q+Vector3(0,.49,0),Vector3(.41,.025,.31),"metal",.004,a)

func _landscape() -> void:
	# Continuous annular terrain: no repeated flat plane, skyline gaps or panorama card.
	# Beyond the collidable perimeter, so these inaccessible slopes need no physics mesh.
	var radii:Array[float]=[9.13,12.0,18.0,27.0,39.0]
	var heights:Array[float]=[0.0,.40,2.8,4.6,6.2]
	for ring in range(radii.size()-1):
		var st:=SurfaceTool.new();st.begin(Mesh.PRIMITIVE_TRIANGLES)
		for i in range(SEGMENTS):
			var a:float=i*TAU/SEGMENTS;var b:float=(i+1)*TAU/SEGMENTS
			var p:=_terrain_point(a,radii[ring],heights[ring],ring)
			var q:=_terrain_point(b,radii[ring],heights[ring],ring)
			var r:=_terrain_point(b,radii[ring+1],heights[ring+1],ring+1)
			var s:=_terrain_point(a,radii[ring+1],heights[ring+1],ring+1)
			_quad(st,p,q,r,s)
		st.generate_normals();st.index();w._add(st.commit(),Vector3.ZERO,"shadow" if ring%2==0 else "metal")
	# Distant small service sheds interrupt the land silhouette without giant structures.
	for i in range(24):
		var a:float=(i+.3)*TAU/24;var y:float=_terrain_point(a,18.0,2.8,2).y
		var p:=_point(a,18.0,y)
		w._box(p+Vector3(0,.45,0),Vector3(1.1,.9,.85),"shadow",.018,a)
		w._box(p+Vector3(0,.93,0),Vector3(1.2,.06,.95),"metal",.008,a)

func _terrain_point(a:float,r:float,y:float,ring:int) -> Vector3:
	var variation:float=sin(a*3+.4)*.48+sin(a*7-1.1)*.28+cos(a*11)*.12
	return _point(a,r,y+variation*minf(float(ring)*.65,1.7))
