"""One frozen model, all fresh final seeds, no post-test parameter selection."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from sim2sim.research.queue import atomic_json
from sim2sim.standalone.candidate import stage,compare
from sim2sim.standalone.cases import write_cases
from sim2sim.standalone.prepare import prepare
from sim2sim.standalone.package import package
from sim2sim.standalone.suite import run
from sim2sim.standalone.replay import shadow
p=Path('results/jolt_learning_20260911').resolve();root=Path.cwd()
frozen=json.loads((p/'final_candidate_freeze.json').read_text());model=root/frozen['source']
assert hashlib.sha256(model.read_bytes()).hexdigest()==frozen['model_sha256']
assert hashlib.sha256((root/frozen['control']).read_bytes()).hexdigest()==frozen['control_sha256']
assert (p/'pipeline_profile_complete.json').exists()
out=p/'final_validation';out.mkdir(exist_ok=False)
old=root/'results/research_20260911/delivery/candidate_models'
base=json.loads((p/'baseline_candidate/summary.json').read_text())
learned=json.loads((p/'candidate_evaluations'/frozen['name']/'suite/summary.json').read_text())
traces={}
for e in base['episodes']:
    traces.setdefault(e['skill'],e['trace'])
traces['roller']=next(e['trace'] for e in learned['episodes'] if e['case']=='roller_brake_3s')
projects={};packages={}
for label,overrides in [('baseline',{}),('candidate',{'roller':model})]:
    bank=out/(label+'_models');stage(old,overrides,bank)
    project=out/(label+'_project')
    subprocess.run(['cp','--reflink=auto','-a',str(p/'runtime_r1'),str(project)],check=True,timeout=60)
    selected=list(traces.values()) if label=='candidate' else [next(e['trace'] for e in base['episodes'] if e['skill']==skill) for skill in traces]
    prepare(bank,project=project,fixture_count=128,real_traces=selected,control_config=root/frozen['control'])
    fixture=json.loads((project/'runtime_assets/self_test.json').read_text())
    assert all(c['real_count']>0 for c in fixture['cases'])
    pkg=package(out/(label+'_package'),project=project)
    with (out/(label+'_self_test.log')).open('w') as log:
        subprocess.run([pkg['executable'],'--headless','--','--self-test'],stdout=log,stderr=subprocess.STDOUT,timeout=45,check=True)
    projects[label]=project;packages[label]=pkg
    atomic_json(out/(label+'_package.json'),pkg)
cases_dir=out/'cases';cases=write_cases(cases_dir,seeds=frozen['final_seeds'])
assert len(cases)==690
atomic_json(out/'case_index.json',{str(Path(c).relative_to(out)):hashlib.sha256(Path(c).read_bytes()).hexdigest() for c in cases})
for label in ['baseline','candidate']:
    result=run(cases,out/(label+'_suite'),workers=4,executable=packages[label]['executable'],timeout=60)
    assert not result['errors']
    atomic_json(out/(label+'_complete.json'),dict(episodes=len(result['episodes']),errors=result['errors']))
comparison=compare(out/'baseline_suite/summary.json',out/'candidate_suite/summary.json',out/'comparison.json')
# The clean environment checks reuse development cases and do not select parameters.
clean_cases=[c for c in sorted((p/'development_cases').glob('*.json')) if json.loads(c.read_text())['seed']==917000]
image='ubuntu@sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254'
clean=run(clean_cases,out/'candidate_clean',workers=4,executable=packages['candidate']['executable'],container_image=image,timeout=90)
assert not clean['errors']
switch=run([p/'contract_switch_case.json'],out/'candidate_switch',workers=1,executable=packages['candidate']['executable'],timeout=60)
assert not switch['errors']
check=shadow(switch['episodes'][0]['trace'],project=projects['candidate'])
atomic_json(out/'switch_shadow.json',check);assert check['passed']
result=dict(completed=True,frozen=frozen,packages=packages,paired_episodes=comparison['paired_episodes'],regressions=comparison['regressions'],final_seeds_consumed=True,clean_episodes=len(clean['episodes']),switch_shadow=check,automatic_promotion=False)
atomic_json(p/'final_validation_complete.json',result)
print('FINAL_VALIDATION_COMPLETE '+json.dumps({k:v for k,v in result.items() if k not in ['packages','frozen']}),flush=True)
