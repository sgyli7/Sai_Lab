extends RefCounted
## Repeated industrial parts: seals, glazed instruments, latches and heat sinks.
## Dimensions describe manufactured assemblies, rather than surface noise.

static func window(w:Node3D,p:Vector3,size:Vector2,angle:float=0.) -> void:
	var out:=Vector3(sin(angle),0,cos(angle))
	w._box(p,Vector3(size.x+.12,size.y+.12,.12),"graphite",.06,angle)
	w._box(p+out*.07,Vector3(size.x,size.y,.035),"glass",.045,angle)
	w._box(p+out*.092+Vector3(0,size.y*.24,0),Vector3(size.x*.85,.022,.012),"blue",.008,angle)
	w._box(p+Vector3(0,size.y*.5+.13,0),Vector3(size.x+.28,.09,.34),"porcelain",.022,angle)

static func hatch(w:Node3D,p:Vector3,size:Vector2,angle:float=0.,color:String="paper") -> void:
	var out:=Vector3(sin(angle),0,cos(angle));var right:=Vector3(cos(angle),0,-sin(angle))
	w._box(p,Vector3(size.x+.055,size.y+.055,.06),"graphite",.07,angle)
	w._box(p+out*.039,Vector3(size.x,size.y,.036),color,.055,angle)
	for side in [-1.,1.]:
		for y in [-.30,.30]:
			var q:Vector3=p+right*size.x*.38*side+Vector3(0,size.y*y,0)+out*.069
			w._box(q,Vector3(.042,.095,.032),"metal",.007,angle)
	w._box(p+right*size.x*.25+out*.078,Vector3(.044,.19,.04),"graphite",.013,angle)
	w._box(p+Vector3(-.06,-size.y*.31,0)+out*.064,Vector3(size.x*.3,.035,.012),"blue",.004,angle)

static func louvers(w:Node3D,p:Vector3,size:Vector2,angle:float=0.) -> void:
	var out:=Vector3(sin(angle),0,cos(angle))
	w._box(p,Vector3(size.x+.12,size.y+.12,.11),"paper",.035,angle)
	w._box(p+out*.064,Vector3(size.x,size.y,.025),"ink",.008,angle)
	var count:int=maxi(3,int(size.y/.11))
	for i in range(count):
		w._box(p+out*.10+Vector3(0,-size.y*.43+i*size.y*.86/(count-1),0),Vector3(size.x*.92,.035,.12),"metal",.008,angle)

static func pipe(w:Node3D,points:Array,r:float,color:String="metal") -> void:
	for i in range(points.size()-1):w._line(points[i],points[i+1],r,color)
	for i in range(1,points.size()-1):
		w._cylinder(points[i],r*1.35,r*.7,"paper",(points[i+1]-points[i]).normalized())

static func tank(w:Node3D,p:Vector3,r:float,h:float) -> void:
	w._cylinder(p+Vector3(0,h*.5,0),r,h,"paper")
	w._dome(p+Vector3(0,h,0),r,r*.45,"porcelain")
	for y in [.16,h-.16]:
		w._cylinder(p+Vector3(0,y,0),r+.032,.095,"metal")
		w._box(p+Vector3(0,y,r+.07),Vector3(.16,.14,.12),"graphite",.015)
	w._cylinder(p+Vector3(0,h*.55,r+.065),r*.23,.075,"signal",Vector3.BACK)
	w._cylinder(p+Vector3(0,h*.55,r+.11),r*.15,.025,"paper",Vector3.BACK)

static func console(w:Node3D,p:Vector3) -> void:
	w._box(p+Vector3(0,.26,0),Vector3(.12,.52,.14),"graphite",.02)
	w._box(p+Vector3(0,.59,0),Vector3(.52,.32,.29),"paper",.065)
	window(w,p+Vector3(-.06,.62,.154),Vector2(.26,.14))
	for y in [.53,.61,.69]:w._cylinder(p+Vector3(.18,y,.168),.018,.025,"yellow" if y==.61 else "signal",Vector3.BACK)
	w._box(p+Vector3(0,.44,.07),Vector3(.56,.05,.45),"metal",.014)

static func ribbon(w:Node3D,p:Vector3,r:float,height:float,segments:int=32) -> void:
	# Faceted panes with discrete mullions create a dark recessed window belt.
	w._cylinder(p,r,height,"ink")
	for i in range(segments):
		var a:float=i*TAU/segments
		w._box(p+Vector3(sin(a)*(r+.012),0,cos(a)*(r+.012)),Vector3(TAU*r/segments*.86,height*.72,.045),"glass",.018,a)
		if i%4==0:
			w._box(p+Vector3(sin(a)*(r+.052),-height*.12,cos(a)*(r+.052)),Vector3(.08,height*.38,.035),"blue",.005,a)

static func radar(w:Node3D,p:Vector3,r:float) -> void:
	w._dish(p,r)
	# Structural radial ribs on the back of the reflector.
	var tilt:=Basis(Vector3.RIGHT,.70)
	for i in range(12):
		var a:float=i*TAU/12.
		var last:=p+tilt*Vector3(0,-.044,0)
		for j in range(1,5):
			var v:float=j*r*.25
			var q:=p+tilt*Vector3(sin(a)*v,.27*v*v/r-.044,cos(a)*v)
			w._line(last,q,.021,"graphite");last=q
	w._cylinder(p-Vector3(0,.34,0),r*.22,.30,"paper",Vector3.RIGHT)
	for side in [-1,1]:w._box(p+Vector3(side*r*.32,-.62,0),Vector3(.13,.8,.36),"paper",.04)
