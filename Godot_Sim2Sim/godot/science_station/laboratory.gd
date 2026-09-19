extends RefCounted
const F=preload("res://science_station/retro_fittings.gd")
## A low pressure laboratory with a cantilevered instrument deck and a real door.
func build(w:Node3D,p:Vector3) -> void:
	# Lower pressure shell: the south-facing entrance remains a physical opening.
	for i in range(32):
		if i in [0,1,2,30,31]:continue
		var a:float=i*TAU/32.
		w._box(p+Vector3(sin(a)*1.50,.91,cos(a)*1.50),Vector3(.32,1.82,.18),"paper",.024,a)
		if i%4==2:
			F.hatch(w,p+Vector3(sin(a)*1.605,.68,cos(a)*1.605),Vector2(.38,.9),a)
	# A machined double flange, deep glazing and a broad flat thermal shield.
	w._frustum(p+Vector3(0,1.90,0),1.64,2.18,.24,"porcelain")
	w._cylinder(p+Vector3(0,2.065,0),2.24,.10,"graphite")
	F.ribbon(w,p+Vector3(0,2.38,0),1.94,.55)
	w._frustum(p+Vector3(0,2.73,0),2.35,2.14,.16,"porcelain")
	w._frustum(p+Vector3(0,2.88,0),2.14,1.83,.14,"paper")
	w._cylinder(p+Vector3(0,2.98,0),1.78,.065,"metal")
	for i in range(16):
		var a:float=i*TAU/16.
		w._line(p+Vector3(sin(a)*1.2,3.019,cos(a)*1.2),p+Vector3(sin(a)*1.77,3.019,cos(a)*1.77),.006,"graphite")
	# Offset rooftop spectrometer: stacked calibration drum, fork and dish.
	w._cylinder(p+Vector3(-.48,3.17,-.38),.66,.36,"paper")
	w._cylinder(p+Vector3(-.48,3.29,-.38),.69,.11,"blue")
	F.radar(w,p+Vector3(-.48,3.91,-.38),.82)
	for x in [.45,.79]:
		w._cylinder(p+Vector3(x,3.10,-.65),.08,.21,"metal")
		w._line(p+Vector3(x,3.18,-.65),p+Vector3(x,4.9+(x-.45)*1.5,-.65),.012,"graphite")
	# Thick integral door collar, with 1.245 m of real clear opening.
	# The wheel policy coasts during reversals; its measured exit envelope fits here.
	for x in [-.77,.77]:
		w._box(p+Vector3(x,.89,1.46),Vector3(.15,1.78,.40),"porcelain",.065)
		w._box(p+Vector3(signf(x)*.65,.89,1.50),Vector3(.055,1.55,.17),"graphite",.02)
	w._box(p+Vector3(0,1.78,1.46),Vector3(1.70,.16,.42),"porcelain",.065)
	w._box(p+Vector3(0,1.89,1.55),Vector3(1.87,.10,.7),"blue",.02)
	w._label("04",p+Vector3(1.31,1.34,.95),100,.0034,"blue",Vector3(0,54,0))
	w._label("SPECTROMETRY",p+Vector3(0,1.79,1.68),40,.00145,"ink")
	# A real bench-height interaction zone, beneath the projecting rigid deck.
	w._box(p+Vector3(-.9,.4,1.8),Vector3(.85,.045,.6),"metal",.012)
	for x in [-1.25,-.55]:
		for z in [1.56,2.04]:w._box(p+Vector3(x,.19,z),Vector3(.038,.38,.038),"graphite",.006)
	F.console(w,p+Vector3(-1.54,0,2.00))
	for i in range(3):
		w._cylinder(p+Vector3(-1.1+i*.19,.48,1.8),.055,.11,"porcelain")
		w._cylinder(p+Vector3(-1.1+i*.19,.542,1.8),.058,.015,"purple")
	# Discrete internal storage visible in the airlock's shadow.
	w._box(p+Vector3(0,.29,-1.13),Vector3(1.2,.58,.35),"graphite",.025)
	for x in [-.31,.31]:
		for y in [.16,.39]:
			F.hatch(w,p+Vector3(x,y,-.945),Vector2(.53,.18),0.,"purple")
	w._box(p+Vector3(0,1.43,-1.12),Vector3(.94,.38,.07),"glass",.03)
	for i in range(5):w._box(p+Vector3(-.32+i*.16,1.43,-1.071),Vector3(.035,.23,.015),"blue",.002)
	# Rear service spine: pressure bottles, connector manifold and radiator.
	for z in [-.65,.18]:
		F.tank(w,p+Vector3(-1.95,.17,z),.27,1.12)
		F.pipe(w,[p+Vector3(-1.95,.24,z),p+Vector3(-2.35,.24,z),p+Vector3(-2.35,1.48,z),p+Vector3(-1.48,1.48,z)],.042)
	F.louvers(w,p+Vector3(1.45,.90,-.72),Vector2(.75,1.05),PI*.64)
	for z in [-.85,.75]:
		w._box(p+Vector3(-2.03,.11,z),Vector3(.95,.22,.52),"graphite",.04)
	w._cabinet(p+Vector3(-2.5,0,2.2),"purple","SPECIMEN / 04")
	w._sign(p+Vector3(3.4,0,.15),"样本处理\nSAMPLE / 04",.7)
