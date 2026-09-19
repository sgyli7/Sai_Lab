extends RefCounted
const F=preload("res://science_station/retro_fittings.gd")
## A prefabricated maintenance capsule; the robot yard is outside its open bay.
func build(w:Node3D) -> void:
	var p:=Vector3(-1.7,0,10.4)
	w._box(p+Vector3(0,1.15,1.02),Vector3(4.7,2.3,.26),"paper",.10)
	for x in [-2.24,2.24]:
		w._box(p+Vector3(x,1.15,0),Vector3(.24,2.3,2.1),"paper",.10)
		w._box(p+Vector3(x,.13,0),Vector3(.35,.26,2.3),"graphite",.03)
	w._box(p+Vector3(0,2.30,0),Vector3(4.8,.22,2.5),"porcelain",.08)
	w._box(p+Vector3(0,2.42,.22),Vector3(4.30,.10,1.55),"metal",.04)
	# Thick cantilever, dark underside, longitudinal ribs, recessed work lights.
	w._box(p+Vector3(0,2.15,-1.24),Vector3(5.03,.22,1.52),"porcelain",.07)
	w._box(p+Vector3(0,2.025,-1.30),Vector3(4.65,.045,1.19),"graphite",.015)
	w._box(p+Vector3(0,2.16,-2.02),Vector3(4.81,.13,.07),"purple",.02)
	for x in [-1.75,-.7,.7,1.75]:
		w._box(p+Vector3(x,1.99,-1.25),Vector3(.065,.07,1.2),"metal",.01)
		w._box(p+Vector3(x,1.947,-1.58),Vector3(.30,.027,.16),"light",.009)
	# Front pressure-door cheek and insulated workshop bulkhead.
	w._box(p+Vector3(1.30,1.06,-1.04),Vector3(1.58,2.12,.24),"paper",.065)
	F.hatch(w,p+Vector3(1.28,1.03,-1.18),Vector2(.95,1.67),PI)
	F.window(w,p+Vector3(1.28,1.40,-1.22),Vector2(.57,.32),PI)
	w._label("07",p+Vector3(1.3,1.96,-1.19),64,.0028,"blue",Vector3(0,180,0))
	w._label("FIELD ENGINEERING",p+Vector3(-.25,2.17,-2.065),42,.0018,"ink",Vector3(0,180,0))
	# Deep utility bay and hard-cased drawers, readable behind the threshold.
	w._box(p+Vector3(-.75,.47,.53),Vector3(1.85,.86,.60),"graphite",.025)
	for x in [-1.30,-.55]:
		for y in [.24,.51,.75]:F.hatch(w,p+Vector3(x,y,.205),Vector2(.64,.20),PI,"purple")
	w._box(p+Vector3(-.78,.93,.49),Vector3(2.05,.07,.80),"metal",.02)
	F.window(w,p+Vector3(-.70,1.49,.855),Vector2(1.46,.50),PI)
	for i in range(3):
		w._box(p+Vector3(-1.4+i*.36,1.035,.43),Vector3(.26,.14,.25),"yellow" if i==1 else "paper",.018)
	# Roof extraction bank is a manufactured cluster with a blue heat shield.
	w._box(p+Vector3(.85,2.70,.33),Vector3(1.7,.48,1.12),"paper",.08)
	F.louvers(w,p+Vector3(.85,2.73,-.255),Vector2(1.32,.29),PI)
	for x in [-1.45,-.98]:
		F.tank(w,p+Vector3(x,2.48,.48),.16,.56)
	F.pipe(w,[p+Vector3(2.40,.3,.50),p+Vector3(2.55,.3,.50),p+Vector3(2.55,2.12,.50),p+Vector3(1.5,2.50,.5)],.075)
	F.louvers(w,p+Vector3(-2.385,1.2,.15),Vector2(1.15,.78),-PI*.5)
	w._cabinet(Vector3(1.6,0,9),"blue","CHARGE / 03")
	w._sign(Vector3(.9,0,8.8),"风口科学站\nWINDPASS / 07",.9)
	# Rear and side elevations have access seals and a dark observation strip too.
	for x in [-1.42,0.,1.42]:
		F.hatch(w,p+Vector3(x,.73,1.17),Vector2(1.18,.88))
		F.window(w,p+Vector3(x,1.74,1.17),Vector2(1.12,.45))
	w._box(p+Vector3(0,.28,1.21),Vector3(4.25,.08,.08),"blue",.012)
	w._label("SERVICE / 07",p+Vector3(1.22,2.32,1.278),48,.0022,"blue")
