from pathlib import Path
import subprocess,json,re,hashlib
r=Path('results/sprint_stop_state_20260912').resolve();project=r/'diagnostic/runtime'
project.parent.mkdir(exist_ok=False)
subprocess.run(['cp','--reflink=auto','-a','results/sprint_joint_gpu_20260912/native_trained_tracking_split/suite/runtime',str(project)],check=True,timeout=60)
brain=re.findall(r'^var (\w+)',(project/'standalone/play_brain.gd').read_text(),re.M)
motion=re.findall(r'^var (\w+)',(project/'standalone/motion_control.gd').read_text(),re.M)
s='''extends "res://standalone/driver.gd"
# [DEBUG-stop-state] Offline diagnosis only; never copied into the game runtime.
const BRAIN_FIELDS = %s
const MOTION_FIELDS = %s
var probe_mode = "warm"
var snapshot_path = ""
var restored = false
var captured = false

func _ready() -> void:
 for arg in OS.get_cmdline_user_args():
  if arg.begins_with("--probe-mode="): probe_mode=arg.trim_prefix("--probe-mode=")
  elif arg.begins_with("--snapshot="): snapshot_path=arg.trim_prefix("--snapshot=")
 super._ready()
 if not ready_to_run: return
 if probe_mode=="cold":
  var file=FileAccess.open(snapshot_path,FileAccess.READ)
  if file==null:
   _fatal("[DEBUG-stop-state] missing snapshot")
   return
  var state=file.get_var()
  file.close()
  _restore_probe(state)
  restored=true

func _fields(obj, names):
 var out={}
 for key in names: out[key]=obj.get(key)
 return out.duplicate(true)

func _set_fields(obj, values):
 for key in values: obj.set(key,values[key])

func _capture_probe():
 var bodies={}
 var measured={}
 for key in _bodies:
  var b=_bodies[key]
  bodies[key]={"transform":b.global_transform,"linvel":b.linear_velocity,"angvel":b.angular_velocity,"sleeping":b.sleeping}
  var p=_g2m(b.global_transform.origin)
  var q=_basis_to_m_quat(b.global_transform.basis)
  var v=_g2m(b.linear_velocity)
  var w=_g2m(b.angular_velocity)
  var k=_g2m(_kin_omega.get(key,Vector3.ZERO))
  measured[key]={"pos":[p.x,p.y,p.z],"quat":[q.w,q.x,q.y,q.z],"linvel":[v.x,v.y,v.z],"angvel":[w.x,w.y,w.z],"kin_omega":[k.x,k.y,k.z]}
 var joints=[]
 var measured_joints=[]
 for j in _joints:
  var fields={}
  for key in ["q_kin_old","qd_kin","q_rebake","tau","axis_dot"]:
   if j.has(key): fields[key]=j[key]
  joints.append(fields)
  var parent_w=Vector3.ZERO
  if j.parent is RigidBody3D: parent_w=j.parent.angular_velocity
  measured_joints.append({"act_index":j.act_index,"q":_joint_q(j),"qd_kin":_joint_qd(j),"qd_solver":(j.child.angular_velocity-parent_w).dot(_axis_godot(j))})
 var snapshot={"bodies":bodies,"joints":joints,"brain":_fields(brain,BRAIN_FIELDS),"motion":_fields(motion,MOTION_FIELDS),"last_action":last_action,"heading":heading,"session":session.duplicate(true),"t":_t,"ctrl":_ctrl,"kin_basis":_kin_basis.duplicate(true),"kin_omega":_kin_omega.duplicate(true),"raw":local_reply.duplicate(true)}
 var file=FileAccess.open(snapshot_path,FileAccess.WRITE)
 file.store_var(snapshot)
 file.close()
 file=FileAccess.open(snapshot_path+".json",FileAccess.WRITE)
 file.store_string(JSON.stringify({"bodies":measured,"joints":measured_joints,"t":_t,"steps":session.steps,"raw":local_reply,"brain":snapshot.brain,"motion":snapshot.motion}))
 file.close()
 print("[DEBUG-stop-state] captured "+str(session.steps))

func _restore_probe(s):
 var trace_path=session.trace_path
 var begun=session.started_usec
 session=s.session.duplicate(true)
 session.trace_path=trace_path
 session.started_usec=begun
 get_tree().set_meta("microduck_session",session)
 _set_fields(brain,s.brain)
 _set_fields(motion,s.motion)
 last_action=s.last_action.duplicate()
 heading=s.heading.duplicate()
 _t=s.t
 _ctrl=s.ctrl.duplicate()
 for key in s.bodies:
  var b=_bodies[key]
  var v=s.bodies[key]
  b.global_transform=v.transform
  b.force_update_transform()
  b.linear_velocity=v.linvel
  b.angular_velocity=v.angvel
 _snapshot_velocities()
 _freeze(false)
 for i in range(_joints.size()):
  for key in s.joints[i]: _joints[i][key]=s.joints[i][key]
 _kin_basis=s.kin_basis.duplicate(true)
 _kin_omega=s.kin_omega.duplicate(true)
 _kin_after_tick=false
 _refresh_mj_basis()
 _remaining=0
 _pending_send=false
 _research_contact_events.clear()
 _send_state("step")
 print("[DEBUG-stop-state] restored "+str(session.steps))

func _decide(held: Array,taps: Array,order: Array,elapsed: float) -> bool:
 if int(session.steps)==450 and not captured:
  if probe_mode=="warm": _capture_probe()
  captured=true
 return super._decide(held,taps,order,elapsed)
'''%(json.dumps(brain),json.dumps(motion))
(project/'standalone/stop_state_probe.gd').write_text(s)
f=project/'standalone/main.tscn';f.write_text(f.read_text().replace('res://standalone/driver.gd','res://standalone/stop_state_probe.gd'))
(r/'hypotheses.json').write_text(json.dumps(dict(reproduction='3/3 original native stop failures',hypotheses=[dict(id='cache',prediction='Cold Jolt restart changes outcome despite identical initial visible observation and physical link states'),dict(id='projection',prediction='Observed FD/EMA velocities differ from solver velocities; projecting visible velocities is not full dynamical state equivalence'),dict(id='domain',prediction='Cold Jolt still fails while GPU projected restart passes; contact/constraint or sensor dynamics remain')],project=str(project),instrumentation_sha256=hashlib.sha256(s.encode()).hexdigest(),production_modified=False),indent=2)+'\n')
