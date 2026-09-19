"""Fixed-budget GPU information ablation on existing native Jolt trajectories.

Run under sim2sim.research.budget. No policy learning or simulator is started.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from sim2sim.research.observability import ARMS, WIDTH, episode_features, arm_features
from sim2sim.research.queue import atomic_json


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'results/sprint_target_gpu_20260912'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def protocol(epochs=40):
    return dict(version=2, architecture=[WIDTH, 128, 128, 3], activation='SiLU',
                arms=list(ARMS), seeds=[997101, 997102], epochs=epochs, batch_size=4096,
                adam_lr=.001, weight_decay=0, early_stopping=False,
                precision='CUDA float32, TF32 disabled', stride=4, history_taps=[1,2,4,8,16],
                targets=['delta_com_vx_current_yaw', 'delta_com_vy_current_yaw',
                         'delta_world_vertical_angular_velocity'], horizon_s=.02,
                target_units=['m/s', 'm/s', 'rad/s'],
                normalization='Train-only mean and std, each std floored at 1e-6',
                split='Whole reset seeds across all policy and noise copies; train fresh seeds except seed%8==2 (validation). Exclude 927001 entirely from train/validation. Test development seeds 927000..927015 only.',
                subsets=['all', 'moving', 'transition'],
                transition='First second after semantic held-key set changes, excluding initial idle',
                gate=dict(required_subsets=['moving', 'transition'], min_rmse_reduction=.10,
                          required_comparators=['immediate','duplicate_current'],
                          max_component_regression=.02, learner_seeds='both',
                          bootstrap='2000 paired reset-seed clusters, 95% percentile interval lower improvement > 0'),
                scope='Information diagnostic in native target physics, not causal proof of policy improvement or final acceptance',
                validation='Every 10 epochs, learning curves only; fixed declared endpoint, never chosen by validation/test')


def sources():
    result = []
    for arm in ('train', 'train_warm', 'train_initial_teacher'):
        for iteration in sorted((OLD/arm).glob('iteration_*')):
            if not (iteration/'completed.json').exists():
                raise RuntimeError('Incomplete source iteration')
            for complete in sorted((iteration/'rollouts').glob('episode_*/completed.json')):
                episode = json.loads(complete.read_text())
                case = json.loads(Path(episode['case']).read_text())
                seed = case['seed']
                if seed == 927001:
                    continue
                if not 955208 <= seed <= 955299:
                    raise ValueError(f'Unexpected learning reset seed {seed}')
                result.append(dict(split='validation' if seed % 8 == 2 else 'train',
                                   seed=seed, source=arm+'/'+iteration.name,
                                   trace=episode['trace'], case_path=episode['case'], case=case['case']))
    suites = [ROOT/'results/sprint_stop_state_20260912/native_joint_fd']
    suites += [OLD/name for name in ('native_006','native_012','native_warm_006','native_warm_012',
                                    'native_initial_teacher_006','native_initial_teacher_012')]
    for suite in suites:
        summary = json.loads((suite/'suite/summary.json').read_text())
        for episode in summary['episodes']:
            if not 927000 <= episode['seed'] <= 927015 or episode['case'].endswith('_ordinary'):
                continue
            case = json.loads(Path(episode['case_path']).read_text())
            if case.get('ordinary_control'):
                raise ValueError('Repeated ordinary control entered test corpus')
            result.append(dict(split='test', seed=episode['seed'], source=suite.name,
                               trace=episode['trace'], case_path=episode['case_path'], case=case['case']))
    groups = {split: {e['seed'] for e in result if e['split']==split} for split in ('train','validation','test')}
    for a, b in (('train','validation'),('train','test'),('validation','test')):
        if groups[a] & groups[b]:
            raise ValueError('Reset seed leakage')
    return result, groups


def extract_one(source):
    raw = Path(source['trace']).read_bytes()
    trace = json.loads(raw)
    summary = trace['summary']
    if summary['error'] or summary['resets'] or summary['switches'] or summary['first_fall'] is not None:
        raise ValueError('Invalid source episode')
    is_training = 'training_exploration' in summary
    if is_training != (source['split'] != 'test'):
        raise ValueError('Exploration label inconsistent with split')
    rows = [r for r in trace['rows'] if 'obs' in r]
    data = episode_features(rows)
    audit = dict(source, trace_sha256=hashlib.sha256(raw).hexdigest(),
                 case_sha256=sha(source['case_path']), samples=len(data['y']), decisions=len(rows),
                 models=summary['models'])
    return audit, data


def extract(output):
    if (output/'dataset.json').exists():
        raise RuntimeError('Completed dataset exists; do not silently overwrite')
    entries, groups = sources()
    pieces = {s: [] for s in groups}
    audits = []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, (audit, data) in enumerate(pool.map(extract_one, entries, chunksize=2)):
            data['seed'] = np.full(len(data['y']), audit['seed'], dtype=np.int64)
            data['episode'] = np.full(len(data['y']), i, dtype=np.int32)
            pieces[audit['split']].append(data)
            audits.append(audit)
            if (i+1) % 100 == 0:
                print(json.dumps(dict(extracted=i+1, total=len(entries), elapsed_s=time.monotonic()-start)), flush=True)
    counts = {}
    for split, parts in pieces.items():
        folder = output/'data'/split
        folder.mkdir(parents=True, exist_ok=False)
        for key in parts[0]:
            np.save(folder/(key+'.npy'), np.concatenate([x[key] for x in parts]))
        counts[split] = dict(episodes=len(parts), samples=sum(len(x['y']) for x in parts),
                             reset_seeds=sorted(groups[split]))
    atomic_json(output/'dataset.json', dict(splits=counts, episodes=audits, elapsed_s=time.monotonic()-start,
                files={str(p.relative_to(output)): sha(p) for p in sorted((output/'data').rglob('*.npy'))}))
    print(json.dumps(dict(dataset_completed=True, counts=counts, elapsed_s=time.monotonic()-start)), flush=True)


def load_split(output, split):
    return {p.stem: np.load(p) for p in (output/'data'/split).glob('*.npy')}


def learn(output):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    p = json.loads((output/'protocol.json').read_text())['protocol']
    if p != protocol(p['epochs']):
        raise RuntimeError('Protocol changed after freeze')
    frozen = json.loads((output/'protocol.json').read_text())
    for path, digest in frozen['sources'].items():
        if sha(ROOT/path) != digest:
            raise RuntimeError('Source changed after protocol freeze: '+path)
    data = {s: load_split(output,s) for s in ('train','validation','test')}
    xt, yt = data['train']['x'], data['train']['y']
    xm = xt.mean(0, dtype=np.float64).astype(np.float32)
    xs = np.maximum(xt.std(0, dtype=np.float64), 1e-6).astype(np.float32)
    ym = yt.mean(0, dtype=np.float64).astype(np.float32)
    ys = np.maximum(yt.std(0, dtype=np.float64), 1e-6).astype(np.float32)
    np.savez(output/'normalizers.npz', x_mean=xm, x_std=xs, y_mean=ym, y_std=ys)
    for d in data.values():
        d['x'] = (d['x']-xm)/xs
        d['normalized_y'] = (d['y']-ym)/ys
    start = time.monotonic()
    records = []
    for seed in p['seeds']:
        for arm in p['arms']:
            folder = output/'learners'/f'{arm}_{seed}'
            folder.mkdir(parents=True, exist_ok=False)
            torch.manual_seed(seed)
            model = torch.nn.Sequential(torch.nn.Linear(WIDTH,128), torch.nn.SiLU(),
                                        torch.nn.Linear(128,128), torch.nn.SiLU(),
                                        torch.nn.Linear(128,3)).cuda()
            initial_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            optimizer = torch.optim.Adam(model.parameters(), lr=p['adam_lr'])
            xs_gpu = {s: torch.as_tensor(arm_features(d['x'],arm), device='cuda') for s,d in data.items()}
            ys_gpu = {s: torch.as_tensor(d['normalized_y'], device='cuda') for s,d in data.items()}
            generator = torch.Generator(device='cuda').manual_seed(seed+1000)
            history = []
            torch.cuda.synchronize()
            train_start = time.monotonic()
            for epoch in range(1,p['epochs']+1):
                total = torch.zeros((),device='cuda')
                model.train()
                for ix in torch.randperm(len(xt),device='cuda',generator=generator).split(p['batch_size']):
                    prediction = model(xs_gpu['train'][ix])
                    loss = torch.mean((prediction-ys_gpu['train'][ix])**2)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    total += loss.detach()*len(ix)
                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        validation = torch.cat([model(x) for x in xs_gpu['validation'].split(8192)])
                        error = torch.mean((validation-ys_gpu['validation'])**2).item()
                    history.append(dict(epoch=epoch,train_mse=total.item()/len(xt),validation_mse=error))
                    print(json.dumps(dict(arm=arm,seed=seed,**history[-1])),flush=True)
            torch.cuda.synchronize()
            learning_s = time.monotonic()-train_start
            with torch.no_grad():
                prediction = torch.cat([model(x) for x in xs_gpu['test'].split(8192)]).cpu().numpy()
            np.save(folder/'test_prediction.npy',prediction*ys+ym)
            torch.save(dict(model=model.state_dict(), protocol=p, seed=seed, arm=arm),folder/'model.pt')
            record = dict(arm=arm, seed=seed, device=str(next(model.parameters()).device),
                          gpu_name=torch.cuda.get_device_name(), initial_state_sha256=initial_hash,
                          learning_s=learning_s, curves=history,
                          prediction_sha256=sha(folder/'test_prediction.npy'),model_sha256=sha(folder/'model.pt'))
            atomic_json(folder/'completed.json',record)
            records.append(record)
            del xs_gpu, ys_gpu, model, optimizer
    atomic_json(output/'learning.json',dict(completed=True,records=records,elapsed_s=time.monotonic()-start,
                torch_version=torch.__version__, cuda_version=torch.version.cuda,
                normalizers_sha256=sha(output/'normalizers.npz')))


def analyze(output):
    data = load_split(output,'test')
    norm = np.load(output/'normalizers.npz')
    y, scale = data['y'], norm['y_std']
    reset_seeds = np.unique(data['seed'])
    subsets = dict(all=np.ones(len(y),dtype=bool),moving=data['moving'],transition=data['transition'])
    results = {}

    def metrics(prediction):
        error = ((prediction-y)/scale)**2
        raw_error = (prediction-y)**2
        out = {}
        for name, mask in subsets.items():
            sums = np.array([error[mask & (data['seed']==seed)].sum(0,dtype=np.float64) for seed in reset_seeds])
            counts = np.array([(mask & (data['seed']==seed)).sum() for seed in reset_seeds])
            out[name] = dict(normalized_rmse=float(np.sqrt(error[mask].mean())),
                             component_rmse=np.sqrt(raw_error[mask].mean(0)).tolist(),
                             normalized_component_rmse=np.sqrt(error[mask].mean(0)).tolist(),
                             seed_sse=sums.tolist(),seed_counts=counts.tolist(),samples=int(mask.sum()))
        return out

    results['persistence'] = metrics(np.zeros_like(y))
    rng = np.random.default_rng(998100)
    bootstrap = rng.integers(0,len(reset_seeds),(2000,len(reset_seeds)))
    decisions = []
    p = json.loads((output/'protocol.json').read_text())['protocol']
    for seed in p['seeds']:
        for arm in ARMS:
            key = f'{arm}_{seed}'
            results[key] = metrics(np.load(output/'learners'/key/'test_prediction.npy'))
        for arm in ('state','contacts','history'):
            checks = {}
            for comparator in ('immediate','duplicate_current'):
                for subset in ('moving','transition'):
                    baseline = results[f'{comparator}_{seed}'][subset]
                    candidate = results[f'{arm}_{seed}'][subset]
                    reduction = 1-candidate['normalized_rmse']/baseline['normalized_rmse']
                    comp = np.array(candidate['component_rmse'])/baseline['component_rmse']-1
                    base_sse = np.array(baseline['seed_sse']).sum(1)
                    cand_sse = np.array(candidate['seed_sse']).sum(1)
                    boot_reduction = 1-np.sqrt(cand_sse[bootstrap].sum(1)/base_sse[bootstrap].sum(1))
                    ci = np.quantile(boot_reduction,[.025,.975])
                    checks[comparator+'/'+subset] = dict(rmse_reduction=float(reduction),component_regression=comp.tolist(),
                                          bootstrap_95_ci=ci.tolist(),
                                          passed=bool(reduction>=.1 and max(comp)<=.02 and ci[0]>0))
            decisions.append(dict(seed=seed,arm=arm,checks=checks,passed=all(x['passed'] for x in checks.values())))
    selected = [arm for arm in ('state','contacts','history') if all(d['passed'] for d in decisions if d['arm']==arm)]
    atomic_json(output/'analysis.json',dict(results=results,decisions=decisions,information_gate_passed=selected,
                test_reset_seeds=reset_seeds.tolist(),scope=p['scope']))
    print(json.dumps(dict(information_gate_passed=selected,decisions=decisions)),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','extract','learn','analyze'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,choices=(40,640),default=40,
                        help='Used only when freezing a new protocol')
    args = parser.parse_args()
    if args.stage in ('extract','learn'):
        session = Path(os.environ.get('SIM2SIM_ACTIVE_BUDGET_DIR','/nonexistent'))
        if not session.exists() or not args.output.resolve().is_relative_to(session.resolve()):
            raise RuntimeError('Use the active-budget supervisor')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage == 'freeze':
        path = args.output/'protocol.json'
        if path.exists():
            raise RuntimeError('Protocol already frozen')
        atomic_json(path,dict(protocol=protocol(args.epochs),created_unix=time.time(),
                    sources={str(p.relative_to(ROOT)):sha(p) for p in
                             (Path(__file__),ROOT/'src/sim2sim/research/observability.py')}))
    else:
        globals()[args.stage](args.output)


if __name__ == '__main__':
    main()
