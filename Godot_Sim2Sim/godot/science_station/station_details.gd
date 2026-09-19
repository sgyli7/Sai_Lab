extends RefCounted
## Close details for the field relay.

func relay(w:Node3D,p:Vector3) -> void:
	# Same small building, but all visible elevations describe a useful instrument.
	for x in [-.45,.1]:
		w._box(p+Vector3(x,.88,.611),Vector3(.40,.35,.04),"graphite",.035)
		w._box(p+Vector3(x,.90,.638),Vector3(.32,.25,.018),"blue",.022)
		w._box(p+Vector3(x,1.08,.68),Vector3(.48,.035,.22),"paper",.008)
	w._box(p+Vector3(.55,.60,.61),Vector3(.19,.65,.04),"paper",.012)
	w._box(p+Vector3(.56,.60,.638),Vector3(.025,.11,.016),"graphite",.003)
	for y in [.20,.28,.36]:w._box(p+Vector3(-.45,y,.62),Vector3(.31,.027,.03),"metal",.005)
	for z in [-.33,.30]:
		w._cylinder(p+Vector3(.80,.63,z),.06,.82,"metal")
		w._cylinder(p+Vector3(.80,.91,z),.09,.08,"paper")
	w._box(p+Vector3(-.76,.48,0),Vector3(.20,.58,.65),"blue",.025)
	for i in range(4):w._box(p+Vector3(-.87,.3+i*.11,0),Vector3(.035,.04,.46),"graphite",.008)
	w._line(p+Vector3(.42,1.36,-.3),p+Vector3(.42,2.50,-.3),.012,"graphite")
	w._label("FIELD RELAY",p+Vector3(-.1,.49,.63),36,.0012,"blue")
