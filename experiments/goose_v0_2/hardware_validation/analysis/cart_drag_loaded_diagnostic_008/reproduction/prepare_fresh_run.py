"""I/O-only preparation; requires a NEW bounded authorization. Never runs physics."""
import argparse,ast,json,shutil,datetime,xml.etree.ElementTree as ET
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,choices=range(1,7),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-utc',required=True,help='ISO8601 future UTC deadline for separately authorized run');a=p.parse_args()
d=Path(__file__).resolve().parents[1];root=d.parents[1];c=json.loads((d/'reproduction_contract.json').read_text());out=a.output.resolve();snap=d/'source_snapshot';v=c['variants'][a.attempt-1]
if out.exists() or out.is_relative_to(root) or out.is_relative_to(Path(c['original_root'])):raise SystemExit('Require new output outside archive/raw')
deadline=datetime.datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00'));assert deadline.tzinfo and deadline>datetime.datetime.now(datetime.timezone.utc)
out.mkdir(parents=True);(out/'scene').mkdir()
replacements={c['original_root']:str(out),'/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots/robots/goose_v0_2/models/physics_v0_2_20261018_04':str(root/c['frozen04']),'/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_probe_006/full_body_readiness':str(root/c['readiness']),'/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_whole_body_007/l1_bias_proposal.json':str(root/c['bias']),'/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots/experiments/goose_v0_2/cart_drag_loaded_diagnostic_008/order.json':str(d/'order/order.json')}
for q in snap.iterdir():
 if q.is_file() and q.suffix in ['.py','.json']:
  if q.suffix=='.py':
   s=q.read_text()
   for old,new in replacements.items():s=s.replace(old,new)
   old='datetime.datetime(2026,10,18,15,18,10,tzinfo=datetime.timezone.utc)'; new=f'datetime.datetime({deadline.year},{deadline.month},{deadline.day},{deadline.hour},{deadline.minute},{deadline.second},tzinfo=datetime.timezone.utc)';s=s.replace(old,new);ast.parse(s);(out/q.name).write_text(s)
  else:shutil.copyfile(q,out/q.name)
for name in ['physical_contract.json','actuator_contract.json']:shutil.copyfile(snap/'scene'/name,out/'scene'/name)
x=ET.parse(snap/'scene/robot.xml');x.getroot().find('compiler').set('meshdir',str(root))
for m in x.getroot().findall('./asset/mesh'):m.set('file',str(root/c['mesh_dependencies'][m.get('file')]['package_path']))
x.write(out/'scene/robot.xml');command=['python',str(out/v['harness']),str(a.attempt),str(out/v['program']) if v['program'] else '1.0'];print(json.dumps(dict(status='PREPARED_ONLY',future_command=command,no_integrator_execution=True)))
