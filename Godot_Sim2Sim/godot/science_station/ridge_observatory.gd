extends RefCounted
## A remote weather observatory, designed in the round for the western skyline.
## Large shaded crown, asymmetric service wing, and a legible structural spine.
func build(w:Node3D,p:Vector3) -> void:
	# A ground-level foundation on the graded site; the building has no rock pedestal.
	w._box(p+Vector3(0,.08,0),Vector3(6.6,.32,4.3),"paper",.04)
	w._box(p+Vector3(0,.235,0),Vector3(6.4,.035,4.1),"metal",.01)
	for i in range(3):
		var h:float=.08*(3-i)
		w._box(p+Vector3(-.5,h*.5,2.22+i*.25),Vector3(1.25,h,.30),"paper",.012)
	# The lower skirt settles into a narrow insulated trunk.
	w._frustum(p+Vector3(-.5,.77,0),1.78,1.39,1.3,"paper")
	w._frustum(p+Vector3(-.5,2.56,0),1.39,1.13,2.27,"paper")
	w._ring(p+Vector3(-.5,1.47,0),1.38,.032,"blue")
	w._ring(p+Vector3(-.5,.28,0),1.74,.038,"graphite")
	# Four deep window pockets read as an inhabited instrument room.
	for spec in [[-.68,2.45],[.05,2.55],[.68,2.45],[2.3,2.55],[3.05,2.5],[3.8,2.4]]:
		var a:float=spec[0];var y:float=spec[1]
		var q:=p+Vector3(-.5+sin(a)*1.27,y,cos(a)*1.27)
		w._box(q,Vector3(.38,.75,.055),"graphite",.10,a)
		w._box(q+Vector3(sin(a)*.033,0,cos(a)*.033),Vector3(.28,.62,.025),"blue",.07,a)
		w._box(q+Vector3(0,.42,0),Vector3(.46,.07,.25),"paper",.02,a)
	# A spreading ribbed transition carries a broad, thin observation crown.
	var crown:=p+Vector3(-.5,3.95,0)
	w._frustum(crown,1.13,2.94,.63,"metal")
	for i in range(18):
		var a:float=i*TAU/18.
		w._beam(crown+Vector3(sin(a)*1.15,-.29,cos(a)*1.15),crown+Vector3(sin(a)*2.86,.31,cos(a)*2.86),.04,.09,"graphite")
	w._frustum(crown+Vector3(0,.41,0),3.07,3.17,.20,"paper")
	w._cylinder(crown+Vector3(0,.73,0),2.77,.43,"graphite")
	for i in range(28):
		var a:float=i*TAU/28.
		var q:=crown+Vector3(sin(a)*2.78,.75,cos(a)*2.78)
		w._box(q,Vector3(.48,.29,.035),"blue" if i%7 else "yellow",.018,a)
	w._frustum(crown+Vector3(0,1.02,0),3.25,2.81,.14,"paper")
	w._dome(crown+Vector3(0,1.08,0),2.81,.53,"paper")
	w._ring(crown+Vector3(0,1.075,0),3.12,.024,"graphite")
	# Meridians follow the cap, terminating before its quiet centre.
	for i in range(16):
		var a:float=i*TAU/16.
		var last:=crown+Vector3(sin(a)*2.795,1.10,cos(a)*2.795)
		for j in range(1,6):
			var t:float=j*.20
			var q:=crown+Vector3(sin(a)*2.81*cos(t),1.08+.53*sin(t)+.009,cos(a)*2.81*cos(t))
			w._line(last,q,.008,"strata");last=q
	w._cylinder(crown+Vector3(.35,1.65,0),.24,.20,"metal")
	for spec in [[.35,3.55,.018],[-.30,2.55,.012],[1.37,2.75,.013]]:
		w._line(crown+Vector3(spec[0],1.4,0),crown+Vector3(spec[0],spec[1],0),spec[2],"graphite")
	w._line(crown+Vector3(.35,2.7,0),crown+Vector3(1.3,2.7,0),.018,"graphite")
	# Offset equipment wing: a low horizontal foil to the observation drum.
	w._box(p+Vector3(2.03,.87,-.30),Vector3(1.65,1.38,2.52),"paper",.13)
	w._box(p+Vector3(2.03,1.62,-.30),Vector3(1.93,.12,2.78),"purple",.028)
	w._box(p+Vector3(2.03,1.02,1.),Vector3(1.20,.67,.055),"graphite",.05)
	for i in range(5):w._box(p+Vector3(2.03,.77+i*.12,1.045),Vector3(1.08,.045,.10),"metal",.008)
	w._cylinder(p+Vector3(2.24,1.96,-.74),.29,.55,"metal")
	w._dome(p+Vector3(2.24,2.23,-.74),.38,.18,"paper")
	for z in [-1.78,1.78]:
		for x in [-3.,-1.5,1.5,3.]:w._line(p+Vector3(x,.24,z),p+Vector3(x,.97,z),.021,"graphite")
		if z<0.:w._line(p+Vector3(-3.,.98,z),p+Vector3(3.,.98,z),.024,"metal")
		else:
			w._line(p+Vector3(-3.,.98,z),p+Vector3(-1.5,.98,z),.024,"metal")
			w._line(p+Vector3(1.5,.98,z),p+Vector3(3.,.98,z),.024,"metal")
	for x in [-2.4,-1.78]:
		w._cylinder(p+Vector3(x,1.,-1.45),.24,1.54,"paper")
		for y in [.50,1.46]:w._ring(p+Vector3(x,y,-1.45),.26,.020,"blue")
		var bend:=p+Vector3(x,-.18,-2.1)
		var socket:=Vector3(bend.x,p.y+.02,bend.z)
		w._line(p+Vector3(x,.28,-1.45),bend,.048,"graphite")
		w._line(bend,socket,.048,"graphite")
		w._cylinder(socket,.085,.12,"blue")
	w._label("WIND / 09",p+Vector3(-.5,.83,1.63),64,.0045,"blue")
