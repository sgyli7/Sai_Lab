extends RefCounted
const VisualProfile = preload("res://atelier/visual_profile.gd")
## Individually placed mechanical detail: connectors, panel seams and printed labels.
var w: Node3D

func build(workshop: Node3D) -> void:
	w=workshop
	_manifold();_bench_radio();_charger_details();_painted_labels();_floor_hatch();_crate_details();_wall_conduit();_cooling_blower();_bench_latched_panels()

func _ring(p: Vector3,r: float,color: String="ink",thickness: float=.001) -> void:
	for i in range(40):
		var a:=i*TAU/40;var b:=(i+1)*TAU/40
		w._line(p+Vector3(cos(a)*r,sin(a)*r,0),p+Vector3(cos(b)*r,sin(b)*r,0),thickness,color)

func _gauge(p: Vector3,r: float,label: String) -> void:
	w._cylinder(p,r,.015,"graphite",Vector3.BACK)
	w._cylinder(p+Vector3(0,0,.009),r*.88,.008,"porcelain",Vector3.BACK)
	_ring(p+Vector3(0,0,.015),r*.76,"ink",.0006)
	for i in range(11):
		var a:=deg_to_rad(-30+i*24.0)
		w._line(p+Vector3(cos(a),sin(a),0)*r*.62+Vector3(0,0,.016),p+Vector3(cos(a),sin(a),0)*r*.73+Vector3(0,0,.016),.0006)
	w._line(p+Vector3(0,0,.018),p+Vector3(-r*.36,r*.37,.018),.0009)
	w._cylinder(p+Vector3(0,0,.019),r*.085,.003,"yellow",Vector3.BACK)
	w._label(label,p+Vector3(0,-r*.32,.018),24,r*.006,"ink")

func _manifold() -> void:
	var p:=Vector3(-1.55,0,-.70)
	# Mount the pressure gauge to the cabinet instead of leaving its dial in air.
	w._line(p+Vector3(-.073,.507,.155),p+Vector3(-.073,.574,.155),.006,"metal")
	w._cylinder(p+Vector3(-.073,.519,.155),.013,.012,"graphite")
	w._cylinder(p+Vector3(-.073,.555,.155),.009,.011,"metal")
	_gauge(p+Vector3(-.073,.60,.155),.034,"kPa")
	for i in range(2):
		var z:=-.10+i*.22
		for y in [.57,.69]:
			w._cylinder(p+Vector3(.08,y,z),.068,.013,"metal")
		w._line(p+Vector3(.081,.75,z),p+Vector3(.23,.75,z),.009,"metal")
		w._line(p+Vector3(.23,.75,z),p+Vector3(.23,.30,z),.009,"metal")
		w._cylinder(p+Vector3(.23,.47,z),.015,.035,"graphite")
		w._cylinder(p+Vector3(.254,.47,z),.029,.008,"purple",Vector3.RIGHT)
	# Functional side grille with a border and open slats.
	w._panel(p+Vector3(0,.265,.283),Vector2(.19,.033),"graphite")
	w._label("DC 24V",p+Vector3(.02,.145,.300),26,.00042,"ink")
	for i in range(3):
		w._cylinder(p+Vector3(-.07+i*.07,.46,.28),.005,.006,"yellow" if i==0 else "metal",Vector3.BACK)

func _bench_radio() -> void:
	var p:=Vector3(-.97,.462,-1.17)
	w._box(p,Vector3(.19,.078,.082),"porcelain",.008)
	w._panel(p+Vector3(0,0,.044),Vector2(.176,.064),"graphite")
	for i in range(8):
		w._line(p+Vector3(-.069+i*.011,-.020,.061),p+Vector3(-.069+i*.011,.020,.061),.001,"metal")
	w._cylinder(p+Vector3(.052,0,.066),.017,.012,"purple",Vector3.BACK)
	w._line(p+Vector3(.06,.035,-.02),p+Vector3(.10,.19,-.02),.002,"metal")
	w._label("FM",p+Vector3(.053,-.025,.065),20,.00025,"porcelain")
	# Coiled cable from bench meter to the inspection mat.
	var previous:=Vector3(-.16,.431,-1.16)
	for i in range(1,45):
		var t:=i/44.0
		var point:=Vector3(-.16-.32*t,.436+sin(t*TAU*7)*.003,-1.16+.09*sin(t*PI))
		w._line(previous,point,.0015,"rubber");previous=point
	# Measuring scale and a neatly folded service note.
	w._box(Vector3(-.57,.434,-1.07),Vector3(.22,.002,.025),"porcelain",.0005)
	for i in range(22):
		w._line(Vector3(-.674+i*.01,.436,-1.081),Vector3(-.674+i*.01,.436,-1.071 if i%5 else -1.062),.00045)
	w._box(Vector3(-.04,.430,-1.10),Vector3(.10,.001,.13),"porcelain",.0002,.1)
	if VisualProfile.value("MD_SERVICE_NOTE")=="1":
		var page:=MeshInstance3D.new()
		var plane:=PlaneMesh.new();plane.size=Vector2(.096,.126)
		var printing:=ShaderMaterial.new();printing.shader=load("res://atelier/printed_paper.gdshader")
		printing.set_shader_parameter("printing",load("res://atelier/graphics/service_note.svg"))
		page.mesh=plane;page.material_override=printing
		page.position=Vector3(-.04,.4312,-1.10);page.rotation.y=.1
		page.cast_shadow=GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		w.add_child(page)
		w._label("MD / SERVICE",Vector3(-.045,.432,-1.149),18,.00025,"graphite",Vector3(-90,5.72958,0))
	else:
		w._label("MD / SERVICE\nCHECK 01",Vector3(-.04,.432,-1.10),20,.0003,"graphite",Vector3(-90,0,0))

func _charger_details() -> void:
	var p:=Vector3(.74,0,-1.34)
	# Recessed connector collars, finger grips, mounting points and instruction plates.
	for x in [-.075,0,.075]:
		_ring(p+Vector3(x,.20,.211),.017,"ink",.001)
		for i in range(4):
			var a:=i*TAU/4
			w._cylinder(p+Vector3(x+cos(a)*.009,.20+sin(a)*.009,.213),.0018,.003,"metal",Vector3.BACK)
	w._label("AUX     CHARGE     DATA",p+Vector3(0,.245,.185),24,.00026,"graphite")
	w._label("ISOLATE BEFORE SERVICE",p+Vector3(0,.135,.189),23,.0003,"graphite")
	for x in [-.143,.143]:
		w._box(p+Vector3(x,.07,.19),Vector3(.03,.04,.011),"yellow",.003)
	# Analog service display next to the digital charge screen.
	_gauge(Vector3(1.25,.60,-1.50),.067,"PRESSURE")
	w._panel(Vector3(1.25,.40,-1.51),Vector2(.19,.15),"paper")
	for i in range(3):
		w._cylinder(Vector3(1.19+i*.06,.43,-1.487),.012,.013,"yellow" if i==0 else "graphite",Vector3.BACK)
	w._label("AIR / 06",Vector3(1.25,.36,-1.486),25,.0004)

func _painted_labels() -> void:
	# Plaques live on surfaces. All text is original workshop fiction.
	w._panel(Vector3(-1.19,.46,-1.512),Vector2(.20,.10),"yellow")
	w._label("02\nPOWER UNIT",Vector3(-1.19,.46,-1.49),28,.00040)
	w._label("MICRODUCK",Vector3(-.84,.265,-1.016),30,.00037,"graphite")
	w._label("PRECISION TOOLS",Vector3(-.84,.13,-1.012),22,.00033,"graphite")
	w._label("01",Vector3(-1.38,.44,-.982),50,.0005,"porcelain")
	# Small serial tags with a simple barcode.
	for p in [Vector3(2.62 if w.open_route else 1.40,.30,1.11 if w.open_route else .67),Vector3(-1.41,.30,.79),Vector3(1.41,.41,-.565)]:
		p.z+=.0015
		w._box(p,Vector3(.085,.035,.002),"porcelain",.001)
		for i in range(15):
			var width:=.001 if i%3 else .0022
			w._box(p+Vector3(-.032+i*.0045,0,.002),Vector3(width,.021,.001),"graphite",.0001)

func _floor_hatch() -> void:
	var p:=Vector3(-.86,.002,.86)
	w._box(p,Vector3(.34,.001,.39),"shadow",.0004)
	w._box(p+Vector3(0,.001,0),Vector3(.328,.001,.378),"floor",.0004)
	for x in [-.135,.135]:
		for z in [-.16,.16]:
			w._cylinder(p+Vector3(x,.002,z),.006,.001,"metal")
	for x in [-.10,.10]:
		w._box(p+Vector3(x,.002,0),Vector3(.022,.001,.077),"graphite",.0005)
	w._label("ACCESS / 07",p+Vector3(0,.003,.12),24,.0005,"shadow",Vector3(-90,0,0))
	# Sparse hairline wear, positioned on a few tiles rather than screen-space noise.
	for item in [Vector3(-.92,.002,.15),Vector3(.63,.002,-.31),Vector3(.36,.002,.85)]:
		w._line(item,item+Vector3(.045,0,.015),.00045,"shadow")
		w._line(item+Vector3(.036,0,.012),item+Vector3(.052,0,.004),.00035,"shadow")

func _crate_details() -> void:
	for p in [Vector3(-1.44,0,.64),Vector3(2.70 if w.open_route else 1.48,0,.90 if w.open_route else .46)]:
		for x in [-.13,.13]:
			w._line(p+Vector3(x,.045,.206),p+Vector3(x,.24,.206),.001,"ink")
		# Latch hinge pins visibly run through both halves of the latch.
		for x in [-.085,.085]:
			w._cylinder(p+Vector3(x,.267,.21),.004,.04,"graphite",Vector3.RIGHT)
	# Small wall-mounted hose reel, with three broad concentric loops.
	var p:=Vector3(-1.69,.30,.78 if w.open_route else .22)
	w._panel(p,Vector2(.14,.19),"graphite")
	for r in [.048,.059,.070]:_ring(p+Vector3(0,0,.03),r,"rubber",.004)
	w._cylinder(p+Vector3(0,0,.034),.022,.05,"purple",Vector3.BACK)

func _wall_conduit() -> void:
	# Narrow service runs fill the wall junctions, with joints and retention clips.
	for x in [.29,.34]:
		w._line(Vector3(x,.10,-1.50),Vector3(x,.72,-1.50),.005,"metal")
		for y in [.20,.45,.65]:w._box(Vector3(x,y,-1.505),Vector3(.022,.015,.02),"graphite",.002)
	w._panel(Vector3(.315,.17,-1.485),Vector2(.115,.10),"paper")
	w._label("24 V",Vector3(.315,.17,-1.463),23,.00037)

func _cooling_blower() -> void:
	# Visible side of the power module: removable curved cover, blower and captive screws.
	var p:=Vector3(-1.388,.265,-.73)
	w._box(p,Vector3(.012,.36,.36),"ink",.016)
	w._box(p+Vector3(.009,0,0),Vector3(.012,.35,.35),"graphite",.015)
	for y in [-.15,.15]:
		for z in [-.15,.15]:
			w._cylinder(p+Vector3(.02,y,z),.007,.005,"metal",Vector3.RIGHT)
	var center:=p+Vector3(.025,.025,-.025)
	w._cylinder(center,.103,.012,"metal",Vector3.RIGHT)
	w._cylinder(center+Vector3(.01,0,0),.093,.013,"rubber",Vector3.RIGHT)
	var blades:=SurfaceTool.new();blades.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in range(8):
		var a:=i*TAU/8
		var pos:=Vector3(0,cos(a)*.064,sin(a)*.064)
		blades.append_from(w._rounded_box(Vector3(.007,.023,.071),.003),0,Transform3D(Basis(Vector3.RIGHT,a+.30),pos))
	var rotor:=MeshInstance3D.new();rotor.name="CoolingRotor"
	rotor.mesh=blades.commit();rotor.material_override=w.materials.metal
	rotor.position=center+Vector3(.018,0,0);w.add_child(rotor);w.rotors.append(rotor)
	w._cylinder(center+Vector3(.023,0,0),.027,.017,"purple",Vector3.RIGHT)
	w._cylinder(center+Vector3(.034,0,0),.010,.006,"graphite",Vector3.RIGHT)
	for r in [.092,.099]:
		for i in range(48):
			var a:=i*TAU/48;var b:=(i+1)*TAU/48
			w._line(center+Vector3(.021,cos(a)*r,sin(a)*r),center+Vector3(.021,cos(b)*r,sin(b)*r),.001)
	for i in range(8):
		var a:=i*TAU/8
		w._cylinder(center+Vector3(.020,cos(a)*.10,sin(a)*.10),.0033,.006,"graphite",Vector3.RIGHT)
	# Bottom vents are cut visually into a recessed panel, avoiding a flat black side.
	for i in range(7):
		w._box(p+Vector3(.019,-.13,-.105+i*.033),Vector3(.006,.018,.020),"ink",.002)
	w._label("COOLING / 02",p+Vector3(.023,.145,0),26,.00064,"porcelain",Vector3(0,90,0))

func _bench_latched_panels() -> void:
	# Thin nested service panels with interrupted seams rather than extra bulky boxes.
	var p:=Vector3(-.20,.48,-1.46)
	w._panel(p,Vector2(.31,.18),"paper")
	w._panel(p+Vector3(0,0,.02),Vector2(.28,.15),"graphite")
	for x in [-.112,.112]:
		w._box(p+Vector3(x,0,.037),Vector3(.017,.08,.012),"metal",.003)
		w._cylinder(p+Vector3(x,.02,.045),.005,.003,"ink",Vector3.BACK)
	w._label("MOTOR CONTROL\n14 AXES",p+Vector3(0,0,.043),26,.00065,"porcelain")
	# A small segmented articulated inspection arm on the end of the bench.
	var foot:=Vector3(-1.0,.44,-1.30)
	w._cylinder(foot,.035,.02,"graphite")
	var joints: Array[Vector3]=[foot+Vector3(0,.04,0),foot+Vector3(-.065,.18,0),foot+Vector3(.07,.25,0)]
	for j in range(2):
		w._line(joints[j],joints[j+1],.010,"metal")
		w._cylinder(joints[j],.019,.021,"graphite",Vector3.BACK)
		w._cylinder(joints[j]+Vector3(0,0,.013),.010,.006,"yellow",Vector3.BACK)
	w._box(joints[2],Vector3(.10,.027,.05),"purple",.006)
	w._box(joints[2]+Vector3(0,-.016,0),Vector3(.079,.008,.03),"light",.002)
