extends RefCounted
## Purposeful asymmetry: weather intake A and optical instrument B.
func build(w:Node3D,p:Vector3,h:float,r:float,id:String) -> void:
	# Suspended plumbing stays above the usable ground-level passage.
	for x in [-.31,.31]:
		for z in [-.24,.24]:
			w._cylinder(p+Vector3(x,1.25,z),.095,.53,"metal")
			w._cylinder(p+Vector3(x,1.03,z),.112,.06,"yellow")
			w._line(p+Vector3(x,1.0,z),p+Vector3(x*.45,.88,z),.026,"graphite")
	w._cylinder(p+Vector3(0,1.36,0),.30,.39,"graphite")
	# Cable trays follow one rear leg, away from the two cross passages.
	var b:=p+Vector3(-r*.68,1.90,-r*.68)
	w._box(b,Vector3(.16,.22,.14),"metal",.02)
	for shift in [-.035,.035]:
		var last:=b+Vector3(shift,0,0)
		for q in [b+Vector3(shift,-.30,-.16),b+Vector3(shift-.2,-.66,-.34),b+Vector3(shift-.33,-1.35,-.42)]:
			w._line(last,q,.017,"purple");last=q
	# Hatches and seams form a few clusters with large calm areas between them.
	for spec in [[-.42,2.7,.43,.72],[.55,3.3,.25,.48],[-.74,h*.69,.32,.82]]:
		var a:float=spec[0];var y:float=spec[1]
		var rr:float=lerpf(r,r*.87,clampf((y-1.8)/(h-2.45),0.,1.))+.018
		var q:=p+Vector3(sin(a)*rr,y,cos(a)*rr)
		w._box(q,Vector3(spec[2],spec[3],.035),"metal",.022,a)
		w._box(q+Vector3(sin(a)*.022,0,cos(a)*.022),Vector3(spec[2]-.045,spec[3]-.055,.012),"paper",.016,a)
		for side in [-1,1]:
			w._box(q+Vector3(cos(a)*spec[2]*.29*side,spec[3]*.30,.042),Vector3(.025,.030,.012),"graphite",.003,a)
	# A deep round optical window is readable from a distance.
	var y:float=h*.67
	var rr:float=lerpf(r,r*.87,(y-1.8)/(h-2.45))
	var eye:=p+Vector3(0,y,rr+.035)
	w._cylinder(eye,.34,.075,"metal",Vector3.FORWARD)
	w._cylinder(eye+Vector3(0,0,.048),.29,.035,"blue",Vector3.FORWARD)
	w._cylinder(eye+Vector3(0,0,.068),.20,.012,"graphite",Vector3.FORWARD)
	w._box(eye+Vector3(-.062,.10,.078),Vector3(.15,.035,.012),"paper",.007)
	var F=load("res://science_station/retro_fittings.gd")
	if id=="02":
		# Broad inhabited scanner crown with a dark continuous instrument belt.
		var crown:=p+Vector3(0,h*.79,0)
		w._frustum(crown,r*.91,r*1.48,.40,"metal")
		for i in range(16):
			var a:float=i*TAU/16.
			w._beam(crown+Vector3(sin(a)*r*.94,-.17,cos(a)*r*.94),crown+Vector3(sin(a)*r*1.45,.20,cos(a)*r*1.45),.055,.12,"graphite")
		w._frustum(crown+Vector3(0,.28,0),r*1.50,r*1.53,.16,"porcelain")
		F.ribbon(w,crown+Vector3(0,.65,0),r*1.36,.55,32)
		w._frustum(crown+Vector3(0,1.02,0),r*1.59,r*1.42,.18,"porcelain")
		w._frustum(crown+Vector3(0,1.20,0),r*1.42,r*1.23,.18,"paper")
		# Offset telemetry blade is carried by a visible fork, not perched on a pole.
		var q:=p+Vector3(-r*1.09,h+.14,-.4)
		w._beam(p+Vector3(-r*.62,h-.6,-.4),q,.16,.27,"metal")
		w._box(q+Vector3(0,.62,0),Vector3(1.0,1.64,.29),"porcelain",.09,-.30)
		w._box(q+Vector3(0,.62,.18),Vector3(.81,1.37,.07),"glass",.035,-.30)
		for j in range(9):w._box(q+Vector3(0,.08+j*.14,.23),Vector3(.72,.012,.02),"metal",.003,-.30)
		w._line(p+Vector3(r*.92,h-.10,0),p+Vector3(r*.92,h+2.10,0),.020,"graphite")
	else:
		# The shorter tower has a projecting atmospheric intake and a scan drum.
		var crown:=p+Vector3(0,h-.87,0)
		w._cylinder(crown,r*1.15,.18,"metal")
		F.ribbon(w,crown+Vector3(0,.38,0),r*1.04,.53,24)
		w._frustum(crown+Vector3(0,.73,0),r*1.25,r*1.10,.17,"porcelain")
		var q:=p+Vector3(-r*.92,h*.58,r*.43)
		F.louvers(w,q,Vector2(.68,1.35),-.7)
		w._beam(p+Vector3(-r*.85,h-.10,0),p+Vector3(-r*1.8,h+.22,0),.10,.16,"metal")
		for j in range(6):w._cylinder(p+Vector3(-r*1.8,h+.26+j*.12,0),.23,.04,"paper")
		w._line(p+Vector3(-r*1.8,h+.22,0),p+Vector3(-r*1.8,h+1.12,0),.03,"graphite")
	# Compression seals, connected actuator housings and an external service spine.
	for y2 in [2.12,2.34]:w._cylinder(p+Vector3(0,y2,0),r*.985,.08,"metal")
	for i in range(6):
		var a:float=i*TAU/6.+.3
		var q:=p+Vector3(sin(a)*r*.995,2.60,cos(a)*r*.995)
		w._box(q,Vector3(.27,.48,.18),"porcelain",.05,a)
		w._box(q+Vector3(sin(a)*.11,0,cos(a)*.11),Vector3(.12,.24,.08),"graphite",.025,a)
	for x in [-.30,.30]:
		F.pipe(w,[p+Vector3(x,2.0,-r*.96),p+Vector3(x,h*.7,-r*.94),p+Vector3(x,h*.77,-r*.73)],.035)
	# Hip joints and exposed piston pairs carry the large instrument mass.
	for i in range(4):
		var a:float=PI*.25+i*PI*.5
		var out:=Vector3(cos(a),0,sin(a))
		var foot:=p+out*(r+.65)+Vector3(0,.20,0)
		var neck:=p+out*r*.72+Vector3(0,1.65,0)
		var joint:=foot.lerp(neck,.58)+out*.17
		w._cylinder(joint,.20,.16,"graphite",out)
		w._cylinder(joint+out*.10,.13,.055,"porcelain",out)
		w._cylinder(joint+out*.135,.055,.018,"blue",out)
		w._line(foot+out*.15+Vector3(0,.07,0),joint,.055,"graphite")
		w._line(foot+out*.15+Vector3(0,.07,0),foot.lerp(joint,.57)+out*.05,.079,"paper")
	# Legible registration is painted on the body; tiny data stays near service panels.
	var tag_y:float=h*.47
	var tag_r:float=lerpf(r,r*.87,(tag_y-1.8)/(h-2.45))
	w._label(id,p+Vector3(0,tag_y,tag_r+.031),120,.0044,"blue")
	w._label("ATMOSPHERIC SURVEY" if id=="01" else "DEEP FIELD ARRAY",p+Vector3(0,tag_y-.36,tag_r+.044),40,.0012,"graphite")
