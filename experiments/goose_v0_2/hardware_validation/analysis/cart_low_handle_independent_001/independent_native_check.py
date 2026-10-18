"""Independent R2 cart audit: saved CAD, compilation and FK only; never calls mj_step."""
from pathlib import Path
import json, hashlib, collections, time
import numpy as np
import build123d as bd
import mujoco as mj
from scipy.spatial import ConvexHull
W=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots')
L=Path('/home/ethan/Projects/TempWorktree/Sai_Lab/goose_v02_hardware_validation/experiments/goose_v0_2/hardware_validation')
E=W/'robots/goose_v0_2/evidence/low_profile_stock_cart_02'; A=L/'assets/low_profile_stock_cart_02'; O=L/'analysis/cart_low_handle_independent_001'; R=L/'inputs/physics_v0_2_20261018_04'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
def save(n,d): (O/n).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def par(m,d): return m*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
def mat(q):
 r=np.zeros(9);mj.mju_quat2Mat(r,np.asarray(q));return r.reshape(3,3)
start=time.monotonic(); p=W/'robots/goose_v0_2/cad/exports/low_profile_stock_cart_02/goose_low_profile_stock_cart_02.step'; native=bd.import_step(p)
screen=load(E/'geometry_mass_screen.json'); parts={r['name']:r for r in screen['parts']}; param=load(W/'robots/goose_v0_2/cad/source/low_profile_stock_cart_02/parameters.json')
print('native imported',flush=True)
# Independent native shape volumes, COMs and complete inertia, not just producer flags.
rows=[]; shapes={}
for child in native.children:
 c=np.array(tuple(child.center(bd.CenterOf.MASS)));v=child.volume
 matches=[n for n,q in parts.items() if abs(q['volume_mm3']-v)<1e-4 and np.max(abs(c-q['COM_mm']))<1e-5]
 assert len(matches)==1,(child.label,matches)
 assert matches[0] not in shapes
 shapes[matches[0]]=child
assert len(shapes)==122
tote=bd.Box(600,400,150).translate((325,0,256.5))
for name,s in shapes.items():
 q=parts[name]; rho=param['materials'][q['material']]['density_kg_m3']*1e-9; v=s.volume; c=np.array(tuple(s.center(bd.CenterOf.MASS))); I=np.array(s.matrix_of_inertia)*rho*1e-6
 common=s.intersect(tote); overlap=0 if common is None else common.volume
 rows.append(dict(name=name,solids=len(s.solids()),positive_volume=v>0,valid=s.is_valid,volume_error_mm3=abs(v-q['volume_mm3']),mass_error_kg=abs(v*rho-q['mass_kg']),COM_error_mm=float(np.max(abs(c-q['COM_mm']))),fullI_error_kg_m2=float(np.max(abs(I-q['full_inertia_about_COM_kg_m2']))),tote_overlap_mm3=float(overlap)))
M=sum(q['mass_kg'] for q in parts.values()); cm=sum(q['mass_kg']*np.array(q['COM_mm']) for q in parts.values())/M
native_report=dict(step_sha256=sha(p),part_count=len(shapes),solid_count=len(native.solids()),valid=native.is_valid,bounds_mm=[list(native.bounding_box().min),list(native.bounding_box().max)],CAD_mass_kg=M,CAD_COM_mm=cm.tolist(),reserve_kg=screen['unmodelled_reserve_kg'],estimated_empty_mass_kg=M+screen['unmodelled_reserve_kg'],max_volume_error_mm3=max(r['volume_error_mm3'] for r in rows),max_COM_error_mm=max(r['COM_error_mm'] for r in rows),max_fullI_error_kg_m2=max(r['fullI_error_kg_m2'] for r in rows),positive_tote_overlaps=[r for r in rows if r['tote_overlap_mm3']>.001],all_positive_valid=all(r['positive_volume'] and r['valid'] and r['solids']==1 for r in rows),gusset_bounds={n:[list(s.bounding_box().min),list(s.bounding_box().max)] for n,s in shapes.items() if n.startswith('mast_base_gusset')},parts=rows)
save('independent_native_geometry.json',native_report);print('native checks complete',flush=True)
