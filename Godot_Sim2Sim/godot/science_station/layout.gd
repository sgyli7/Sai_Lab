extends RefCounted
## Metres in Godot's Y-up frame. These anchors also define safe reset pads.
const TITLE := "风口科学站"
const CHECKPOINTS := {
	"service": {"title":"服务小院", "position":Vector3(0,0,7)},
	"samples": {"title":"样本处理站", "position":Vector3(-5,0,1)},
	"berth": {"title":"设备泊位", "position":Vector3(5,0,5)}
}
const PROPS := [Vector2(.30,7),Vector2(.66,7.08),Vector2(-4.70,1),Vector2(-4.34,1.08),Vector2(5.30,5),Vector2(5.66,5.08)]
const BERTH := Rect2(3,-8,20,12)
const VIEWS := {
	"arrival":[Vector3(4.,1.3,10.8),Vector3(-3.,2.5,-9.),61.],
	"overview":[Vector3(-24.0,12.0,22.0),Vector3(-1.0,1.0,-7.0),53.],
	"towers":[Vector3(10.5,2.0,-7.5),Vector3(10.8,4.0,-18.5),64.],
	"samples":[Vector3(-0.8,1.7,5.5),Vector3(-7.6,1.9,-0.5),59.],
	"berth":[Vector3(23.0,4.0,8.0),Vector3(-2.0,3.0,-16.0),60.],
	"hills":[Vector3(-16.0,1.4,-1.0),Vector3(-34.0,3.0,-22.0),58.]
}

static func height_at(x: float,z: float) -> float:
	# Barycentric interpolation of the exact quarter-metre physics triangles.
	var x0:float=floorf(x*4.)*.25;var z0:float=floorf(z*4.)*.25
	var u:float=(x-x0)*4.;var v:float=(z-z0)*4.
	var b:float=_vertex_height(x0+.25,z0);var c:float=_vertex_height(x0,z0+.25)
	if u+v<=1.:return _vertex_height(x0,z0)*(1.-u-v)+b*u+c*v
	return b*(1.-v)+c*(1.-u)+_vertex_height(x0+.25,z0+.25)*(u+v-1.)

static func _bump(t:float) -> float:
	return pow(maxf(0.,1.-t*t),2.)

static func _rect_distance(p:Vector2,c:Vector2,h:Vector2) -> float:
	var q:Vector2=(p-c).abs()-h
	return q.max(Vector2.ZERO).length()+minf(maxf(q.x,q.y),0.)

static func _prepared_distance(x:float,z:float) -> float:
	var p:=Vector2(x,z)
	var d:float=_rect_distance(p,Vector2(0,10),Vector2(4.5,3.8))
	d=minf(d,(p-Vector2(-7.5,-.5)).length()-5.1)
	d=minf(d,_rect_distance(p,Vector2(-3.5,-17.8),Vector2(10.3,7.7)))
	d=minf(d,_rect_distance(p,Vector2(13,-2),Vector2(10.,6.)))
	d=minf(d,(p-Vector2(7,-17)).length()-3.5)
	d=minf(d,(p-Vector2(16,-21)).length()-4.)
	# Interaction/reset pads and the accepted loop retain a level wheel envelope.
	for c in [Vector2(0,7),Vector2(-5,1),Vector2(5,5)]:d=minf(d,(p-c).length()-1.2)
	for edge in [Vector4(0,7,-2,3.8),Vector4(-2,3.8,-5,1),Vector4(-5,1,5,5),Vector4(5,5,0,7)]:
		var a:=Vector2(edge.x,edge.y);var b:=Vector2(edge.z,edge.w);var v:=b-a
		d=minf(d,(p-a-v*clampf((p-a).dot(v)/v.length_squared(),0.,1.)).length()-.85)
	# The circulation spine is a graded saddle through the basin.
	d=minf(d,_rect_distance(p,Vector2(0,-4),Vector2(2.8,18.)))
	return d

static var _vertices:Dictionary={}

static func _vertex_height(x:float,z:float) -> float:
	var key:=Vector2(x,z)
	if not _vertices.has(key):_vertices[key]=_sculpt_height(x,z)
	return _vertices[key]

static func _traverse_vertex(x:float,z:float) -> float:
	return .14*_bump((x+18.)/6.)*_bump((z+3.)/7.)

static func _traverse_height(x:float,z:float) -> float:
	# Preserve the established driving surface when subdividing its triangles.
	# Resampling the analytic hill on a finer grid subtly changes wheel contacts.
	var x0:float=floorf(x*2.)*.5;var z0:float=floorf(z*2.)*.5
	var u:float=(x-x0)*2.;var v:float=(z-z0)*2.
	var b:float=_traverse_vertex(x0+.5,z0);var c:float=_traverse_vertex(x0,z0+.5)
	if u+v<=1.:return _traverse_vertex(x0,z0)*(1.-u-v)+b*u+c*v
	return b*(1.-v)+c*(1.-u)+_traverse_vertex(x0+.5,z0+.5)*(u+v-1.)

static func _sculpt_height(x:float,z:float) -> float:
	# Small relief belongs to the middle/foreground. The accepted distant hills
	# stay in landscape.gd; no surrounding ridge is raised to hide missing detail.
	var base:float=_traverse_height(x,z)
	var grading:float=smoothstep(.10,1.8,_prepared_distance(x,z))
	var west_axis:float=-18.8+1.3*sin(z*.21)
	var west:float=.76*_bump((x-west_axis)/5.3)*_bump((z-5.)/9.)
	var south:float=.53*_bump((z-11.6-1.4*sin(x*.25))/3.8)*_bump((x+10.)/7.5)
	var east:float=.60*_bump((x-20.)/4.2)*_bump((z-10.-.3*(x-20.))/4.3)
	var relief:float=maxf(west,maxf(south,east))
	# Low oblique sediment shelves are carved into the same heightfield.
	# Varying gaps and burial make their edges intermittent, never circular pads.
	var phase:float=x*.78+z*.42+1.35*sin(z*.29)+.62*sin(x*.52-z*.31)+.75*sin(x*.28)*sin(z*.23)
	var beds:float=smoothstep(.08,.48,sin(phase))
	var folds:float=.055*sin(x*.42+z*.27)+.025*sin(x*1.1-z*.51)
	relief=(relief*(.69+.31*beds)+.09*beds+folds)*grading
	# Feather to the unchanged distant terrain at the rectangular mesh boundary.
	var rim:float=smoothstep(0.,2.2,minf(24.-absf(x),minf(z+26.,14.-z)))
	# The west walking traverse keeps its existing gentle grade and clear width.
	var traverse:float=(1.-smoothstep(.60,2.3,absf(z+3.)))*(1.-smoothstep(5.,7.,absf(x+18.)))
	return base+maxf(-.035,relief)*rim*(1.-traverse)

static func surface_normal(x:float,z:float) -> Vector3:
	# Consistent normals at patch borders; positions and Jolt faces stay exact.
	return Vector3(_vertex_height(x-.25,z)-_vertex_height(x+.25,z),.5,
		_vertex_height(x,z-.25)-_vertex_height(x,z+.25)).normalized()
