extends RefCounted
const F=preload("res://science_station/retro_fittings.gd")
## The north terminus: a pressure habitat, receiver dish and external thermal plant.
## The stepped silhouette anchors the plaza without occupying the future berth.
func build(w:Node3D) -> void:
	var p:=Vector3(-6.5,0,-20.5)
	# Articulated lower hull. Structural feet are outside the front apron.
	for x in [-3.55,3.55]:
		for z in [-1.35,1.35]:
			w._box(p+Vector3(x,.18,z),Vector3(.93,.36,.88),"graphite",.06)
			w._beam(p+Vector3(x,.33,z),p+Vector3(x*.9,.94,z*.85),.32,.37,"metal")
	w._box(p+Vector3(0,1.74,0),Vector3(8.1,2.12,3.3),"paper",.19)
	w._box(p+Vector3(0,.77,0),Vector3(8.35,.22,3.50),"metal",.055)
	w._box(p+Vector3(0,2.89,0),Vector3(8.57,.23,3.74),"porcelain",.06)
	# Front instrument gallery: glazing set behind cream window brows.
	for x in [-2.92,-1.56,1.56,2.92]:
		F.window(w,p+Vector3(x,2.16,1.70),Vector2(1.12,.69))
		F.hatch(w,p+Vector3(x,1.21,1.70),Vector2(1.02,.48))
	F.hatch(w,p+Vector3(0,1.75,1.73),Vector2(.94,1.55))
	w._box(p+Vector3(0,.20,2.21),Vector3(1.48,.40,.52),"metal",.04)
	w._box(p+Vector3(0,.50,1.92),Vector3(1.48,.20,.48),"metal",.025)
	w._label("WINDPASS",p+Vector3(-2.42,2.92,1.895),100,.004,"blue")
	w._label("PLANETARY RESEARCH / 07",p+Vector3(1.95,2.92,1.895),46,.0018,"ink")
	# A long, low glazed bridge offsets the round science instruments elsewhere.
	var c:=p+Vector3(-.45,3.10,-.14)
	w._box(c,Vector3(6.60,.24,3.03),"metal",.08)
	w._box(c+Vector3(0,.56,0),Vector3(5.96,.99,2.54),"graphite",.16)
	for x in [-2.28,-1.14,0.,1.14,2.28]:
		F.window(w,c+Vector3(x,.60,1.30),Vector2(1.00,.69))
	for x in [-3.03,3.03]:
		F.window(w,c+Vector3(x,.6,0),Vector2(1.88,.66),signf(x)*PI*.5)
	w._box(c+Vector3(0,1.18,0),Vector3(6.89,.22,3.38),"porcelain",.09)
	w._box(c+Vector3(0,1.34,-.1),Vector3(6.22,.15,2.63),"paper",.06)
	# Fork-mounted deep-space reflector, offset to the west of the bridge.
	w._cylinder(c+Vector3(-1.04,1.71,-.13),.83,.62,"paper")
	w._cylinder(c+Vector3(-1.04,1.93,-.13),.88,.12,"blue")
	F.radar(w,c+Vector3(-1.04,2.96,-.13),2.35)
	for x in [-2.6,-1.3,0.,1.3,2.6]:
		w._line(c+Vector3(x,1.422,-1.12),c+Vector3(x,1.422,1.12),.006,"strata")
	# A rear comms fin and staggered antennas give the horizon a human purpose.
	w._box(p+Vector3(3.12,4.07,-.38),Vector3(1.07,2.1,1.12),"paper",.12)
	F.louvers(w,p+Vector3(3.12,4.12,.205),Vector2(.72,1.20))
	for spec in [[2.84,7.55],[3.30,6.44],[3.57,8.13]]:
		w._line(p+Vector3(spec[0],5.,-.4),p+Vector3(spec[0],spec[1],-.4),.022,"graphite")
	w._line(p+Vector3(3.3,6.1,-.4),p+Vector3(4.3,6.1,-.4),.025,"metal")
	w._cylinder(p+Vector3(4.3,6.33,-.4),.14,.46,"signal")
	# External life-support module, distinct from the glazed control room.
	for z in [-.92,.16,1.24]:F.tank(w,p+Vector3(-4.56,.40,z),.39,1.82)
	F.louvers(w,p+Vector3(4.09,1.66,0),Vector2(2.36,1.30),PI*.5)
	F.pipe(w,[p+Vector3(-4.56,.45,1.24),p+Vector3(-4.56,.45,2.10),p+Vector3(-3.04,.45,2.10),p+Vector3(-3.04,.85,1.75)],.095)
	# A shallow service gantry at the northern edge. Walkable ground remains open.
	for x in [-3.8,3.8]:
		w._beam(p+Vector3(x,2.74,-1.70),p+Vector3(x,3.33,-2.0),.08,.08,"metal")
	w._line(p+Vector3(-3.8,3.33,-2),p+Vector3(3.8,3.33,-2),.035,"graphite")
	_thermal_plant(w,Vector3(-10.7,0,-7.2))

func _thermal_plant(w:Node3D,p:Vector3) -> void:
	w._box(p+Vector3(0,.18,0),Vector3(2.9,.36,1.65),"graphite",.06)
	for x in [-.91,0.,.91]:
		F.tank(w,p+Vector3(x,.39,0),.29,1.88)
		F.pipe(w,[p+Vector3(x,.58,.35),p+Vector3(x,.58,.79),p+Vector3(x,2.51,.79),p+Vector3(x,2.51,0)],.055)
	w._box(p+Vector3(0,2.60,0),Vector3(3.05,.15,1.6),"porcelain",.035)
	w._box(p+Vector3(-1.35,1.55,-.78),Vector3(.22,1.80,.24),"metal",.03)
	w._box(p+Vector3(1.35,1.55,-.78),Vector3(.22,1.80,.24),"metal",.03)
	for i in range(11):
		w._box(p+Vector3(-1.22+i*.244,2.92,-.65),Vector3(.055,.53,.64),"blue",.012)
	w._box(p+Vector3(1.75,.6,.2),Vector3(.58,1.2,.75),"paper",.065)
	F.window(w,p+Vector3(1.75,.84,.597),Vector2(.35,.29))
	w._label("He / 03",p+Vector3(1.75,.43,.59),44,.0015,"blue")
