"""I/O-only preparation. Does not import a simulator or execute a run."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,choices=[1,2,3],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=Path(__file__).resolve().parents[1]; root=d.parents[1]; out=a.output.resolve()
if out.exists() or out.is_relative_to(root):raise SystemExit('Use a new directory outside the dataset')
out.mkdir(parents=True)
src=d/'source_snapshot'/['run.py','run_l1.py','run_diagnostic.py'][a.attempt-1]
lines=src.read_text().splitlines(); found=0
for i,line in enumerate(lines):
 if line.startswith('R=Path(__file__).parent;P='):
  lines[i]="R=Path(__file__).parent;P=Path("+repr(str(root/'inputs/physics_v0_2_20261018_04'))+");F=Path("+repr(str(d/'preparation/full_body_readiness'))+");num=int(sys.argv[1]);gain=float(sys.argv[2]) if len(sys.argv)>2 else 1.;S=R/f'attempt_{num:03d}';S.mkdir(exist_ok=False)";found+=1
assert found==1,'Frozen source header changed'
(out/'run_fresh.py').write_text('\n'.join(lines)+'\n');(out/'l1_bias_proposal.json').write_bytes((d/'source_snapshot/l1_bias_proposal.json').read_bytes())
print('Prepared only. Future explicit execution: python',out/'run_fresh.py',a.attempt,'1.0')
