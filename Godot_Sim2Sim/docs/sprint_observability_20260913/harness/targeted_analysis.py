"""Post-hoc failure-window diagnostic, never used to select a predictor."""
from pathlib import Path
import json
import numpy as np
from sim2sim.research.queue import atomic_json

ROOT=Path.cwd()
SESSION=ROOT/'results/sprint_observability_20260913'
OLD=ROOT/'results/sprint_target_gpu_20260912'
SUITES=[ROOT/'results/sprint_stop_state_20260912/native_joint_fd']+[OLD/n for n in (
    'native_006','native_012','native_warm_006','native_warm_012',
    'native_initial_teacher_006','native_initial_teacher_012')]


def main():
    manifest=json.loads((SESSION/'predictor/dataset.json').read_text())
    folder=SESSION/'predictor/data/test'
    data={p.stem:np.load(p) for p in folder.glob('*.npy') if p.stem!='x'}
    windows=[]
    mask=np.zeros(len(data['y']),bool)
    original=np.zeros_like(mask)
    paired_success=np.zeros_like(mask)
    by_key={(e['source'],e['case'],e['seed']):i for i,e in enumerate(manifest['episodes']) if e['split']=='test'}
    for suite in SUITES:
        summary=json.loads((suite/'suite/summary.json').read_text())
        for episode in summary['episodes']:
            key=(suite.name,episode['case'],episode['seed'])
            if key not in by_key or episode['task_metrics']['success']:
                continue
            case=json.loads(Path(episode['case_path']).read_text())
            for kind in ('turns','straight','stops'):
                for event in episode['task_metrics'][kind]:
                    if event['passed']:
                        continue
                    at=event['at']
                    end=min([s['at'] for s in case['segments'] if s['at']>at]+[case['seconds']])
                    start=at+1 if kind=='turns' else at
                    if kind=='stops': end=min(end,at+3)
                    item=dict(source=suite.name,case=case['case'],seed=case['seed'],kind=kind,start=start,end=end)
                    windows.append(item)
                    window=(data['t']>=start-1e-7)&(data['t']<end-1e-7)
                    member=data['episode']==by_key[key]
                    mask|=member&window
                    if suite.name=='native_joint_fd':original|=member&window
                    # Same course, seed, time in other frozen candidates which
                    # passed the entire task: descriptive pairing only.
                    for other in SUITES:
                        if other==suite:continue
                        records=json.loads((other/'suite/summary.json').read_text())['episodes']
                        match=next((e for e in records if e['case']==case['case'] and e['seed']==case['seed']),None)
                        otherkey=(other.name,case['case'],case['seed'])
                        if match and match['task_metrics']['success'] and otherkey in by_key:
                            paired_success|=(data['episode']==by_key[otherkey])&window
    selections={'known_failed_windows':mask,'original_failed_turn':original,'same_case_seed_successful_model_windows':paired_success}
    result={}
    for experiment in ('predictor','predictor_long'):
        scale=np.load(SESSION/experiment/'normalizers.npz')['y_std']
        result[experiment]={}
        for complete in sorted((SESSION/experiment/'learners').glob('*/completed.json')):
            prediction=np.load(complete.parent/'test_prediction.npy')
            error=(prediction-data['y'])**2
            result[experiment][complete.parent.name]={name:dict(samples=int(s.sum()),
                normalized_rmse=float(np.sqrt((error[s]/scale**2).mean())),
                component_rmse=np.sqrt(error[s].mean(0)).tolist()) for name,s in selections.items()}
    atomic_json(SESSION/'targeted_analysis.json',dict(windows=windows,results=result,
        scope='Post-hoc development diagnostics on fixed endpoints; failed cases selected by existing scorer. Overlapping failure/success windows are unions. No independent causal or policy improvement claim.'))


if __name__=='__main__':main()
