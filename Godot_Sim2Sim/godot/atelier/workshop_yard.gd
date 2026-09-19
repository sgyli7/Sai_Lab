extends RefCounted
## Original neighbourhood scenery. Only peripheral plots are occupied; main routes stay open.
## Every substantial part uses the same mesh for rendering and convex collision.
var w: Node3D

func build(workshop: Node3D) -> void:
	w=workshop
	if OS.get_environment("MD_YARD_DRESSING")=="0":return
	var previous: bool=w.solid_scope
	w.solid_scope=true
	_repair_annex();_dispatch_shed();_pump_corner();_rear_roofs();_spares_rack();_accessories()
	w.solid_scope=false
	if w.visuals_enabled:
		_windows_and_notices();_service_runs();_lamps();_small_stories();_facade_detail()
	w.solid_scope=previous

func _pipe(points: Array[Vector3],radius: float=.018,color: String="metal") -> void:
	for i in range(points.size()-1):w._line(points[i],points[i+1],radius,color)

func _rim(p: Vector3,r: float,color: String="metal",thickness: float=.006) -> void:
	for i in range(32):
		var a: float=i*TAU/32;var b: float=(i+1)*TAU/32
		w._line(p+Vector3(cos(a)*r,sin(a)*r,0),p+Vector3(cos(b)*r,sin(b)*r,0),thickness,color)

func _repair_annex() -> void:
	# A low inhabited workshop, directly behind the existing service wall.
	w._box(Vector3(-.68,.61,-2.38),Vector3(2.50,1.22,.88),"shadow",.025)
	w._box(Vector3(-.68,1.245,-2.38),Vector3(2.65,.06,1.01),"graphite",.012)
	w._box(Vector3(-.68,1.285,-2.39),Vector3(2.59,.026,.96),"metal",.006)
	for x in [-1.87,-.68,.51]:
		w._box(Vector3(x,.65,-1.925),Vector3(.046,1.16,.045),"graphite",.004)
	# Individual sheets and raised laps avoid coplanar overlays.
	for i in range(13):
		w._box(Vector3(-1.85+i*.195,1.304,-2.39),Vector3(.014,.018,.94),"graphite",.003)
	for x in [-1.47,-.28]:
		w._box(Vector3(x,1.07,-1.911),Vector3(.79,.275,.035),"graphite",.008)
		w._box(Vector3(x,1.07,-1.888),Vector3(.72,.223,.016),"screen",.004)
		for dx in [-.24,0,.24]:
			w._box(Vector3(x+dx,1.07,-1.874),Vector3(.016,.22,.015),"metal",.002)
	# Two folded service sacks rest on the roof, with real collision.
	for x in [-.95,-.70]:
		w._box(Vector3(x,1.355,-2.03),Vector3(.16,.10,.14),"paper",.035)
	# Roof ventilator and its rain hat, small enough to read as machinery.
	w._cylinder(Vector3(-1.47,1.40,-2.43),.105,.19,"graphite")
	w._cylinder(Vector3(-1.47,1.505,-2.43),.148,.035,"paper")
	w._cylinder(Vector3(-1.47,1.53,-2.43),.10,.016,"metal")
	w._box(Vector3(-.50,1.37,-2.43),Vector3(.40,.12,.34),"paper",.022)
	for i in range(6):w._box(Vector3(-.65+i*.06,1.40,-2.251),Vector3(.023,.06,.01),"ink",.001)

func _dispatch_shed() -> void:
	# Real open doorway: side jambs, lintel, side walls and a separate rear wall.
	var p:=Vector3(1.97,0,-2.25)
	w._box(p+Vector3(-.62,.59,0),Vector3(.12,1.18,1.0),"paper",.019)
	w._box(p+Vector3(.62,.59,0),Vector3(.12,1.18,1.0),"paper",.019)
	w._box(p+Vector3(0,.59,-.46),Vector3(1.12,1.18,.08),"shadow",.012)
	w._box(p+Vector3(-.385,.45,.46),Vector3(.35,.90,.08),"paper",.014)
	w._box(p+Vector3(.465,.45,.46),Vector3(.19,.90,.08),"paper",.014)
	w._box(p+Vector3(0,1.04,.46),Vector3(1.12,.28,.08),"paper",.014)
	w._box(p+Vector3(0,1.207,0),Vector3(1.44,.055,1.12),"graphite",.012)
	# Rolled-up shutter sits above a .58 m wide, .90 m tall clear opening.
	w._cylinder(p+Vector3(.08,.985,.535),.067,.64,"metal",Vector3.RIGHT)
	for dx in [-.225,.385]:
		w._box(p+Vector3(dx,.455,.515),Vector3(.025,.91,.035),"graphite",.003)
	w._box(p+Vector3(0,1.13,.675),Vector3(1.39,.037,.49),"purple",.010)
	for dx in [-.58,.58]:_pipe([p+Vector3(dx,.84,.53),p+Vector3(dx,1.11,.89)],.013,"graphite")
	# Shelving inside has open space under it and individually collidable packages.
	for dx in [-.42,.42]:
		w._box(p+Vector3(dx,.23,-.29),Vector3(.03,.46,.03),"graphite",.004)
	w._box(p+Vector3(0,.46,-.29),Vector3(.93,.025,.28),"metal",.004)
	w._crate(p+Vector3(-.26,.475,-.27),Vector3(.29,.21,.22),"purple","REPAIR")
	w._crate(p+Vector3(.19,.475,-.27),Vector3(.36,.17,.22),"porcelain","DELIVERY")
	w._crate(p+Vector3(-.37,.005,-.27),Vector3(.20,.20,.21),"yellow","06")
	# Rain gutter / downpipe is supported and has actual collision at duck height.
	_pipe([p+Vector3(.735,1.19,.42),p+Vector3(.735,.11,.42),p+Vector3(.86,.07,.50)],.019,"graphite")
	for y in [.28,.72,1.08]:w._cylinder(p+Vector3(.735,y,.42),.026,.025,"metal")

func _pump_corner() -> void:
	var p:=Vector3(-2.38,0,-1.38)
	# Squat service tank on three feet, instead of a monumental tower.
	for x in [-.16,.16]:
		for z in [-.13,.13]:w._box(p+Vector3(x,.13,z),Vector3(.033,.26,.033),"graphite",.005)
	w._cylinder(p+Vector3(0,.55,0),.24,.67,"metal")
	for y in [.24,.84]:w._cylinder(p+Vector3(0,y,0),.253,.042,"graphite")
	w._cylinder(p+Vector3(0,.905,0),.17,.045,"paper")
	w._cylinder(p+Vector3(0,1.01,0),.041,.18,"graphite")
	w._cylinder(p+Vector3(0,1.12,0),.089,.028,"paper")
	w._panel(p+Vector3(0,.60,.244),Vector2(.20,.16),"porcelain")
	w._label("AIR\nRESERVE 02",p+Vector3(0,.60,.266),27,.00053)
	_pipe([p+Vector3(.18,.30,.09),p+Vector3(.34,.30,.09),p+Vector3(.34,.12,.09),p+Vector3(.47,.12,.09)],.022,"graphite")
	w._cylinder(p+Vector3(.34,.37,.09),.012,.13,"metal")
	w._cylinder(p+Vector3(.34,.44,.09),.056,.016,"yellow")
	# Small electrical cabinet alongside the tank.
	w._box(Vector3(-2.64,.24,-2.10),Vector3(.40,.48,.35),"graphite",.019)
	w._panel(Vector3(-2.64,.27,-1.914),Vector2(.34,.34),"paper")
	for i in range(5):w._box(Vector3(-2.64,.18+i*.025,-1.89),Vector3(.23,.007,.008),"ink",.001)

func _rear_roofs() -> void:
	# A modest staggered back lane closes the horizon without swallowing the yard.
	w._box(Vector3(-.1,.24,-3.40),Vector3(6.25,.48,.12),"shadow",.018)
	for x in [-3.18,-1.12,.96,3.04]:
		w._box(Vector3(x,.32,-3.40),Vector3(.075,.64,.17),"graphite",.005)
	for item in [Vector3(-2.30,.99,-2.97),Vector3(.57,1.18,-3.08),Vector3(2.65,.80,-3.00)]:
		w._box(Vector3(item.x,item.y*.5,item.z),Vector3(.76,item.y,.53),"shadow",.018)
		w._box(Vector3(item.x,item.y+.025,item.z),Vector3(.84,.048,.61),"metal",.008)
		w._cylinder(Vector3(item.x-.16,item.y+.16,item.z),.05,.26,"graphite")
		w._cylinder(Vector3(item.x-.16,item.y+.30,item.z),.082,.032,"paper")
	# Asymmetric side utility fence leaves the existing left access slit unobstructed.
	w._box(Vector3(-2.99,.19,-1.605),Vector3(.09,.38,2.33),"paper",.018)
	for z in [-2.67,-1.5,-.465]:
		w._box(Vector3(-2.99,.32,z),Vector3(.13,.64,.08),"graphite",.008)

func _spares_rack() -> void:
	var p:=Vector3(3.10,0,-.82)
	# A side vignette: deep open rack, spare wheels and sorted parts.
	for x in [-.31,.31]:
		for z in [-.20,.20]:w._box(p+Vector3(x,.41,z),Vector3(.035,.82,.035),"graphite",.005)
	for y in [.07,.37,.80]:w._box(p+Vector3(0,y,0),Vector3(.70,.027,.47),"metal",.005)
	w._box(p+Vector3(0,.45,-.215),Vector3(.61,.69,.025),"shadow",.005)
	w._crate(p+Vector3(-.16,.084,.025),Vector3(.25,.20,.30),"porcelain","BEARINGS")
	w._crate(p+Vector3(.16,.084,.025),Vector3(.25,.16,.30),"yellow","CABLE")
	for x in [-.17,.15]:
		w._cylinder(p+Vector3(x,.545,.01),.146,.15,"rubber",Vector3.BACK)
		w._cylinder(p+Vector3(x,.545,.096),.09,.023,"metal",Vector3.BACK)
		w._cylinder(p+Vector3(x,.545,.113),.026,.013,"ink",Vector3.BACK)
	w._panel(p+Vector3(0,.86,.21),Vector2(.57,.10),"paper")
	w._label("SALVAGE / 再 生",p+Vector3(0,.86,.233),30,.00052)
	# Returned motor: supported by its plinth; not in the main walking/kicking route.
	w._box(Vector3(3.20,.045,.15),Vector3(.42,.09,.31),"graphite",.009)
	w._cylinder(Vector3(3.20,.18,.15),.105,.30,"purple",Vector3.RIGHT)
	w._cylinder(Vector3(3.37,.18,.15),.04,.06,"metal",Vector3.RIGHT)

func _windows_and_notices() -> void:
	# Unevenly lit window panes, sparse paint repairs, individual sill plants.
	for p in [Vector3(-1.71,1.07,-1.860),Vector3(-.52,1.07,-1.860),Vector3(-.04,1.07,-1.860)]:
		w._box(p,Vector3(.19,.174,.006),"light",.002)
		w._line(p+Vector3(-.075,.01,.006),p+Vector3(.06,.06,.006),.0012,"shadow")
	w._box(Vector3(-.87,1.12,-1.88),Vector3(.025,.09,.12),"graphite",.004)
	w._panel(Vector3(-.87,1.12,-1.812),Vector2(.22,.12),"yellow")
	w._label("OPEN\n修 理 中",Vector3(-.87,1.12,-1.789),28,.00054,"graphite")
	w._panel(Vector3(1.54,.69,-1.733),Vector2(.24,.31),"graphite")
	w._box(Vector3(1.54,.71,-1.713),Vector3(.184,.22,.008),"porcelain",.002)
	w._label("COLLECT\n07 : 30\n——\n配 件 领 取",Vector3(1.54,.71,-1.707),25,.00053)
	for x in [-1.23,-.95]:
		w._cylinder(Vector3(x,.933,-1.868),.034,.073,"purple" if x< -1.1 else "paper")
		for j in range(3):
			var start:=Vector3(x,.97,-1.868)
			var end:=start+Vector3((j-1)*.032,.06+j*.01,0)
			w._line(start,end,.002,"shadow")
			w._box(end,Vector3(.023,.008,.014),"shadow",.004,j*.6)
	# Enamel plates deliberately show only a few chips, not noisy texture everywhere.
	for p in [Vector3(1.34,.39,-1.742),Vector3(-1.84,.93,-1.896),Vector3(2.21,.52,-1.742)]:
		w._line(p,p+Vector3(.043,.006,0),.0014,"shadow")
		w._line(p+Vector3(.025,-.013,0),p+Vector3(.052,-.009,0),.0008,"shadow")

func _service_runs() -> void:
	_pipe([Vector3(-2.39,.87,-1.50),Vector3(-2.39,1.19,-1.71),Vector3(-1.97,1.19,-1.80)],.022,"metal")
	for x in [-1.87,.51]:
		_pipe([Vector3(x,.85,-1.875),Vector3(x,1.37,-1.875),Vector3(x+.17,1.37,-2.1)],.013,"metal")
	# Drooping service cable with restrained hanging tags. All safely above robot height.
	var previous:=Vector3(-2.40,1.23,-1.80)
	for i in range(1,33):
		var t:float=i/32.0
		var next:=Vector3(lerpf(-2.40,2.40,t),1.23-.21*sin(t*PI),-1.80)
		w._line(previous,next,.004,"rubber");previous=next
	for x in [-1.99,.87,2.40]:
		w._line(Vector3(x,.75,-1.90),Vector3(x,1.24,-1.80),.008,"graphite")
	for x in [-1.06,.69,1.19]:
		var t:float=(x+2.4)/4.8;var y:float=1.23-.21*sin(t*PI)
		w._box(Vector3(x,y-.035,-1.80),Vector3(.04,.058,.004),"yellow" if x<0 else "paper",.002)
	# Wall-mounted spare belts have hollow centres, expressed with rings.
	for i in range(3):
		var p:=Vector3(-2.972,.24+i*.018,-.85-i*.22)
		# Against the fence, in a plane parallel to YZ.
		for j in range(32):
			var a:float=j*TAU/32;var b:float=(j+1)*TAU/32
			w._line(p+Vector3(.055,cos(a)*.095,sin(a)*.095),p+Vector3(.055,cos(b)*.095,sin(b)*.095),.006,"rubber")

func _lamps() -> void:
	for p in [Vector3(-1.87,1.01,-1.78),Vector3(2.58,.91,-1.70)]:
		w._line(p+Vector3(0,.10,-.09),p+Vector3(0,.10,0),.009,"graphite")
		w._cylinder(p+Vector3(0,.075,0),.055,.025,"graphite")
		w._cylinder(p+Vector3(0,.025,0),.030,.075,"light")
		w._cylinder(p+Vector3(0,-.025,0),.043,.012,"graphite")
		for x in [-.033,.033]:w._line(p+Vector3(x,-.02,0),p+Vector3(x,.07,0),.0025,"metal")
		if w.server!=null:
			var lamp:=OmniLight3D.new();lamp.name="YardLamp";lamp.position=p+Vector3(0,-.045,.04)
			lamp.light_color=Color("ffe4b1");lamp.light_energy=.36;lamp.omni_range=.65
			lamp.shadow_enabled=false;w.add_child(lamp)

func _small_stories() -> void:
	# Tea cup, ledger and a hand-wound reel bring human scale to the old bench.
	w._cylinder(Vector3(-.92,.429,-1.06),.028,.009,"metal")
	w._cylinder(Vector3(-.92,.466,-1.06),.020,.066,"porcelain")
	w._cylinder(Vector3(-.92,.500,-1.06),.015,.0015,"ink")
	_rim(Vector3(-.898,.473,-1.059),.012,"porcelain",.0025)
	w._box(Vector3(-.78,.432,-1.075),Vector3(.093,.016,.10),"shadow",.003,-.14)
	w._box(Vector3(-.78,.442,-1.075),Vector3(.088,.003,.093),"paper",.001,-.14)

func _accessories() -> void:
	# Wall-attached ladder, plant sill and hose spool have real solid supports.
	w._box(Vector3(-1.11,.881,-1.865),Vector3(.51,.027,.18),"graphite",.004)
	for x in [-2.04,-1.81]:
		w._line(Vector3(x,.035,-1.97),Vector3(x,1.39,-2.06),.012,"graphite")
	for j in range(11):
		var t:float=j/10.0
		w._line(Vector3(-2.04,.09+t*1.20,-1.974-t*.080),Vector3(-1.81,.09+t*1.20,-1.974-t*.080),.012,"metal")
	for y in [.28,1.03]:
		w._line(Vector3(-2.04,y,-2.10),Vector3(-2.04,y,-1.99),.012,"graphite")
	var p:=Vector3(-2.61,.15,-.61)
	w._cylinder(p,.12,.29,"purple")
	for y in [-.115,.115]:w._cylinder(p+Vector3(0,y,0),.125,.021,"graphite")
	w._cylinder(p+Vector3(0,.157,0),.115,.015,"metal")
	w._cylinder(p+Vector3(.05,.169,0),.018,.01,"graphite")
	# Horizontal cable spool beside the side rack, with a low cradle.
	p=Vector3(3.00,.16,.43)
	w._box(p+Vector3(0,-.13,0),Vector3(.24,.06,.22),"graphite",.005)
	w._cylinder(p,.105,.15,"rubber",Vector3.BACK)
	for z in [-.086,.086]:w._cylinder(p+Vector3(0,0,z),.138,.018,"paper",Vector3.BACK)
	w._cylinder(p+Vector3(0,0,.105),.025,.023,"purple",Vector3.BACK)

func _facade_detail() -> void:
	# A hand-maintained pressure clock is a small focal point between the windows.
	w._cylinder(Vector3(.36,1.035,-1.891),.020,.09,"graphite",Vector3.BACK)
	var p:=Vector3(.36,1.035,-1.83)
	w._cylinder(p,.115,.035,"graphite",Vector3.BACK)
	w._cylinder(p+Vector3(0,0,.024),.102,.014,"porcelain",Vector3.BACK)
	_rim(p+Vector3(0,0,.034),.093,"metal",.002)
	for i in range(12):
		var a:float=i*TAU/12
		w._line(p+Vector3(cos(a)*.079,sin(a)*.079,.035),p+Vector3(cos(a)*.088,sin(a)*.088,.035),.0015)
	w._line(p+Vector3(0,0,.037),p+Vector3(-.053,.043,.037),.002)
	w._line(p+Vector3(0,0,.039),p+Vector3(.025,.025,.039),.003,"purple")
	w._cylinder(p+Vector3(0,0,.042),.009,.004,"yellow",Vector3.BACK)
	# Window hinges, framing and old screws: larger marks survive playing distance.
	for x in [-1.84,-1.10,-.65,.09]:
		for y in [.99,1.15]:w._box(Vector3(x,y,-1.865),Vector3(.033,.012,.019),"metal",.002)
	for x in [-1.47,-.28]:
		w._box(Vector3(x,1.204,-1.864),Vector3(.85,.020,.060),"paper",.003)
	# Selective plaster joints, leaving the large walls quiet rather than all-over noise.
	for i in range(8):
		var y:float=.13+i*.101
		w._line(Vector3(2.656,y,-2.62),Vector3(2.656,y,-1.89),.0012,"shadow")
		for z in [-2.44,-2.12]:
			var zz:float=z+(i%2)*.11
			w._line(Vector3(2.656,y,zz),Vector3(2.656,y+.10,zz),.001,"shadow")
	# Braced roof patch and a bolted inspection plate on the distant chimney housing.
	w._panel(Vector3(2.65,.55,-2.727),Vector2(.42,.32),"paper")
	for i in range(7):w._box(Vector3(2.65,.455+i*.031,-2.704),Vector3(.30,.009,.008),"graphite",.001)
	w._label("FILTER 09",Vector3(2.65,.67,-2.697),24,.00055)
	# Hanging service pennant, printed on a physical bracket beside the open doorway.
	w._line(Vector3(2.63,1.02,-1.94),Vector3(2.91,1.02,-1.94),.009,"graphite")
	w._line(Vector3(2.89,1.02,-1.94),Vector3(2.89,.88,-1.94),.003,"metal")
	w._panel(Vector3(2.89,.80,-1.94),Vector2(.18,.17),"yellow")
	w._label("06",Vector3(2.89,.80,-1.917),45,.001)
	# Small seams on the returned motor and cable layers on the spool.
	for i in range(5):
		_rim(Vector3(3.00,.16,.36+i*.026),.107,"graphite",.003)
	for i in range(7):
		w._cylinder(Vector3(3.09+i*.036,.18,.15),.107,.008,"graphite",Vector3.RIGHT)
