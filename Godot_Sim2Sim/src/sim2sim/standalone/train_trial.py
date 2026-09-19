"""Run a declared training recipe, then its paired native development evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from sim2sim.paths import sim2sim_root
from sim2sim.research.budget import require_supervision
from sim2sim.research.queue import atomic_json
from .candidate import evaluate


def trial(recipe_path,name,resume=None):
    recipe=json.loads(Path(recipe_path).read_text());session=Path(os.environ['SIM2SIM_RESEARCH_DIR'])
    require_supervision(session);root=sim2sim_root();started=time.time()
    values=dict(root=str(root),session=str(session),name=name)
    def expand(argument):
        for key,value in values.items():argument=argument.replace('{'+key+'}',value)
        return argument
    command=[sys.executable,'-m','sim2sim.research.train','--name',name,
             *[expand(arg) for arg in recipe['training_arguments']]]
    if resume:command.extend(['--resume',resume])
    subprocess.run(command,timeout=recipe['training_timeout_seconds'],check=True)
    run=session/'runs'/name;completion=json.loads((run/'completed.json').read_text())
    if not completion['final_parity']['passed'] or completion['status']=='interrupted_checkpointed':
        raise RuntimeError('Training stopped without a complete, verified candidate')
    # The terminal policy is declared in advance; the generic training reward
    # or an internal best-checkpoint score does not choose the deployment.
    candidate=run/'final.onnx'
    output=session/'candidate_evaluations'/name
    evaluate(recipe['incumbent'].format(**values),recipe['skill'],candidate,
             recipe['cases'].format(**values),output,recipe['baseline'].format(**values),workers=4)
    comparison=json.loads((output/'comparison.json').read_text())
    result=dict(completed=True,recipe=str(Path(recipe_path).resolve()),training=completion,
        candidate=str(candidate),native_comparison=str(output/'comparison.json'),
        eligible_for_quality_review=comparison['eligible_for_quality_review'],
        elapsed_s=time.time()-started,decision='not_promoted; inspect paired quality evidence')
    atomic_json(run/'trial_completed.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('recipe');p.add_argument('--name',required=True)
    p.add_argument('--resume');a=p.parse_args();print(json.dumps(trial(a.recipe,a.name,a.resume)))
