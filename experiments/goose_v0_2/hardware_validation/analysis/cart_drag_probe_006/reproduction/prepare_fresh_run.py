"""Prepare archived006 inputs in a NEW directory; never imports or runs simulation."""
import argparse,ast,hashlib,json,pathlib,shutil,xml.etree.ElementTree as ET
p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,choices=[1,2,3,4],required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
d=pathlib.Path(__file__).resolve().parents[1];root=d.parents[1];c=json.loads((d/'reproduction_contract.json').read_text());v=c['variants'][a.attempt-1];out=a.output.expanduser().resolve()
if out.exists() or out.is_relative_to(root) or out.is_relative_to(pathlib.Path(c['original_root'])):raise SystemExit('Require new output outside archive and original006')
out.mkdir(parents=True,exist_ok=False);snapshot=d/'source_snapshot';shutil.copy2(snapshot/v['plan'],out/'planning_gate.json');(out/'scene').mkdir()
for name in ['physical_contract.json','actuator_contract.json']:shutil.copy2(snapshot/('attempt_%03d'%a.attempt)/'scene'/name,out/'scene'/name)
x=ET.parse(snapshot/('attempt_%03d'%a.attempt)/'scene/robot.xml');x.getroot().find('compiler').set('meshdir',str(root))
for m in x.getroot().findall('./asset/mesh'):
 dep=c['mesh_dependencies'][m.get('file')];q=root/dep['package_path'];assert hashlib.sha256(q.read_bytes()).hexdigest()==dep['sha256'];m.set('file',str(q))
x.write(out/'scene/robot.xml');shutil.copytree(snapshot/'native_pad_review',out/'native_pad_review')
for q in (out/'native_pad_review').glob('*.py'):q.write_text(q.read_text().replace(c['original_root'],str(out)))
s=(snapshot/v['harness']).read_text();line=next(z for z in s.splitlines() if z.startswith('R=Path(__file__).parent;num='));suffix=line.split("plan=json.loads",1)[1]
replacement="R=Path(__file__).parent;num=int(sys.argv[1]);negative=len(sys.argv)>2 and sys.argv[2]=='open';S=R/f'attempt_{num:03d}';S.mkdir(exist_ok=False);P=Path("+repr(str(root/'inputs/physics_v0_2_20261018_04'))+");plan=json.loads"+suffix
s='# I/O-only relocation from preserved006 source; unexecuted by integrator.\n'+s.replace(line,replacement,1);s=s.replace(c['original_root'],str(out));ast.parse(s);(out/'run_fresh.py').write_text(s)
print(json.dumps(dict(status='PREPARED_ONLY_NOT_EXECUTED',attempt=a.attempt,future_command=['python',str(out/'run_fresh.py'),str(a.attempt)]+(['open'] if v['negative_open_jaw'] else []),requires_new_bounded_authorisation=True)))
