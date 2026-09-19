"""Deadline-aware PPO experiments with exact native anchors and honest exports."""
import argparse,copy,hashlib,json,math,os,random,signal,time,traceback
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from .tasks import TASKS,SESSION,conditions,DT
from .world import World
from .models import Policy,Critic,export_policy,parity
from .rewards import Objective,EXTRA_DIM
from .evaluate import run_suite,PROTOCOL_VERSION
from .mirror import OBS_PERM,JOINT_PERM,JOINT_SIGN,observation_sign
from .setup import read_session
from sim2sim.obs import build_obs

_STOP_REQUESTED = False


def request_stop(signum, frame):
    """Finish the current update and publish a complete checkpoint before exit."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = True

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def reference_observations(task):
    """Native MuJoCo states, with failed full trajectories excluded.

    Locomotion references retain stable moving trajectories and idle explicitly;
    no claim that their command tracking satisfies the stricter new protocol.
    """
    root=SESSION/"references"/task.name
    if not (root/"summary.json").is_file():
        root=SESSION/"evaluation_v4"/task.name/"factory_mujoco"
    if not (root/"summary.json").is_file():
        raise FileNotFoundError('Prepare anchor observations with: python -m sim2sim.research.setup '
                                f'references --skill {task.name}')
    summary=json.loads((root/"summary.json").read_text())
    handoff=SESSION/"evaluation_v4_standing"/task.name/"factory_mujoco/summary.json"
    if not handoff.exists():handoff=SESSION/"handoff_probe"/task.name/"factory_mujoco/summary.json"
    if handoff.exists():summary["episodes"].extend(json.loads(handoff.read_text())["episodes"])
    selected=[]
    for e in summary["episodes"]:
        if 'error' in e:continue
        ok=e["success"]
        if task.name in ("standing","walking","roller"):
            ok=not e["fell"] and (e["condition"] in ("default","idle","game_seq") or abs(e["mean_vx"])>.05 or abs(e["mean_wz"])>.1)
        if ok:selected.append(np.load(e["trace"])["obs"])
    if not selected:raise RuntimeError(f"No valid source reference trajectories for {task.name}")
    return torch.from_numpy(np.concatenate(selected))

class Vector:
    def __init__(self,task,num_envs,seed,weights=None,training_conditions=None,entry="reset",roll_starts=0.,reward_params=None,random_commands=0.,time_input_s=0.,heading_input=False,environment_state=None,entry_source=None,entry_bank=None,roller_contract=False,yaw_memory_input=False,state_input='',roller_objective='legacy',task_input='',walking_objective='legacy',motion_settings=None,sprint_composition=None):
        self.task=task;self.worlds=[];self.objectives=[];self.seed=seed
        self.sprint_composition=sprint_composition
        self.rng=np.random.default_rng(seed);self.count=0
        if environment_state is not None:
            self.seed=int(environment_state['seed']);self.count=int(environment_state['count'])
            if environment_state.get('rng') is not None:self.rng.bit_generator.state=copy.deepcopy(environment_state['rng'])
        self.conditions=training_conditions or conditions(task)
        self.entry=entry;self.entry_counts={"reset":0,"standing":0}
        self.entry_bank=None
        if entry_bank:
            if task.name=='walking':
                if walking_objective=='legacy' or entry_source or motion_settings is None:
                    raise ValueError('Walking entry-bank requires sprint feedback training')
                from .sprint_entry import SprintEntryBank
                self.entry_bank=SprintEntryBank(entry_bank)
            elif task.name not in ('roulade','kick_left','kick_right') or task.robot not in ('microduck_ball','microduck_ball_stand_fix') or entry_source:
                raise ValueError('Native entry-bank prefixes require a roll/kick in the ball scene, without --entry-source')
            else:
                from .entry_bank import EntryBank
                self.entry_bank=EntryBank(entry_bank)
        self.random_commands=random_commands
        if random_commands and task.name not in ("walking","roller"):raise ValueError("Random commands require locomotion")
        self.roll_starts_fraction=roll_starts;self.roll_library=None
        if roll_starts:
            if task.name!="roulade":raise ValueError("--roll-starts requires roulade")
            from .curriculum import RollStarts
            self.roll_library=RollStarts();self.entry_counts["midroll"]=0
        try:
            for i in range(num_envs):
                self.worlds.append(World(task,time_input_s=time_input_s,heading_input=heading_input,entry_source=entry_source,roller_contract=roller_contract,yaw_memory_input=yaw_memory_input,state_input=state_input,task_input=task_input,motion_settings=motion_settings))
                self.worlds[-1].sprint_composed=sprint_composition is not None
            self.reset_worlds(range(num_envs))
            self.objectives=[Objective(w,weights,reward_params,roller_objective,walking_objective) for w in self.worlds]
        except BaseException:
            self.close();raise

    def next_seed(self):
        self.count+=1;return self.seed*100000+self.count

    def checkpoint_state(self):
        return dict(seed=self.seed,count=self.count,rng=copy.deepcopy(self.rng.bit_generator.state))

    def reset_worlds(self,indices):
        warm=[];prefixes=[]
        for i in indices:
            w=self.worlds[i];w.reset(self.next_seed(),self.next_condition(i))
            if self.roll_library is not None and self.rng.random()<self.roll_starts_fraction:
                self.roll_library.reset(w,self.rng);self.entry_counts["midroll"]+=1
                continue
            standing=self.entry=="standing" or (self.entry=="mixed" and self.rng.random()<.5)
            if standing:
                if self.entry_bank is not None:
                    name=self.entry_bank.names[int(self.rng.integers(len(self.entry_bank.names)))]
                    prefixes.append((w,name));self.entry_counts[name+'_prefix']=self.entry_counts.get(name+'_prefix',0)+1
                else:
                    teacher,cmd=w.prepare_standing_entry();warm.append((w,teacher,cmd))
            self.entry_counts["standing" if standing else "reset"]+=1
        # Warmups still use real dynamics and source actions, but independent
        # workers advance together instead of serializing fifty round trips.
        for _ in range(50 if warm else 0):
            for w,teacher,cmd in warm:
                obs=build_obs(w.state,w.last,cmd,w.home);w.send(teacher(obs[None])[0])
            for w,_,_ in warm:w.recv()
        for w,_,_ in warm:w.finish_standing_entry()
        if prefixes:
            for step in range(max(len(self.entry_bank.tapes[n]) for _,n in prefixes)):
                active=[(w,n) for w,n in prefixes if step<len(self.entry_bank.tapes[n])]
                for w,n in active:self.entry_bank.send(w,n,step)
                for w,_ in active:w.recv()
            for w,_ in prefixes:
                if self.task.name=='walking':self.entry_bank.finish(w)
                else:w.finish_standing_entry()

    def next_condition(self,i=0):
        if self.random_commands and self.rng.random()<self.random_commands:return "random_seq"
        return self.conditions[int(self.rng.integers(len(self.conditions)))]

    def observations(self):
        actor=np.stack([w.obs() for w in self.worlds])
        critic=np.concatenate([actor,np.stack([r.extra() for r in self.objectives])],axis=1)
        if self.sprint_composition is not None:
            critic[:,actor.shape[1]+20]=self.sprint_composition.mask(self.worlds).numpy() # unused walking touch channel
        return torch.from_numpy(actor),torch.from_numpy(critic)

    def actor_mask(self):
        return torch.ones(len(self.worlds),dtype=torch.bool) if self.sprint_composition is None or self.sprint_composition.learn_all else self.sprint_composition.mask(self.worlds)

    def step(self,actions):
        if self.sprint_composition is not None:actions=self.sprint_composition.actions(self.worlds,actions)
        for w,a in zip(self.worlds,actions):w.send(a)
        reward=[];done=[];timeouts=[];terms=[]
        for w,obj in zip(self.worlds,self.objectives):
            w.recv();r,terminal,detail=obj.compute()
            timeout=w.t>=self.task.seconds-1e-6 and not terminal
            reward.append(r);done.append(terminal or timeout)
            # One-shot maneuvers end naturally at their task deadline. Only
            # continuous control uses artificial time-limit bootstrapping.
            timeouts.append(timeout and self.task.name in ("standing","walking","roller","sitstand"));terms.append(detail)
        terminal_obs=self.observations()[1]
        if not torch.isfinite(terminal_obs).all():
            self.nonfinite_terminal=dict(critic_obs=terminal_obs,done=done,timeouts=timeouts,
                features=[w.features for w in self.worlds],times=[w.t for w in self.worlds],
                conditions=[w.condition for w in self.worlds])
        finished=[i for i,d in enumerate(done) if d]
        self.reset_worlds(finished)
        for i in finished:self.objectives[i].reset()
        return torch.tensor(reward),torch.tensor(done),torch.tensor(timeouts),terminal_obs,terms

    def close(self):
        for w in self.worlds:
            try:w.close()
            except Exception:pass

def rng_state():
    state={"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()}
    if torch.cuda.is_initialized():state['cuda']=torch.cuda.get_rng_state_all()
    return state

def save_checkpoint(path,policy,critic,ao,co,iteration,config,environment=None):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.partial')
    torch.save({"policy":policy.state_dict(),"critic":critic.state_dict(),"actor_optimizer":ao.state_dict(),"critic_optimizer":co.state_dict(),"iteration":iteration,"config":config,"rng":rng_state(),"factory_sha256":policy.anchor.sha256,"protocol":config['protocol'],
        "environment":None if environment is None else environment.checkpoint_state()},temporary)
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    os.replace(temporary,path)

def run(args):
    torch.set_num_threads(args.threads);seed_all(args.seed)
    resume_checkpoint=torch.load(args.resume,weights_only=False,map_location='cpu') if args.resume else None
    if resume_checkpoint is not None:
        for key,default in [('roller_objective','legacy'),('walking_objective','legacy'),('mask_task_state',False),('mask_motion_state',False),('action_basis',''),
                            ('teacher_mode',''),('teacher_replay',None),('teacher_weight',.02),('teacher_samples',384),('walking_episode_seconds',0.)]:
            if getattr(args,key,default)!=resume_checkpoint['config'].get(key,default):
                raise ValueError('Resume cannot change '+key+'; start a separately recorded experiment')
        recorded_gate=resume_checkpoint['config'].get('time_gate','')
        if not args.time_gate:args.time_gate=recorded_gate
        if args.time_gate!=recorded_gate:raise ValueError('Resume cannot change the actor time gate')
        recorded_command_gate=resume_checkpoint['config'].get('command_gate','')
        if not args.command_gate:args.command_gate=recorded_command_gate
        if args.command_gate!=recorded_command_gate:raise ValueError('Resume cannot change the command gate')
        if args.skill=='walking':
            # Check prefix provenance before creating any simulator processes.
            from .sprint_entry import SprintEntryBank
            fingerprints=SprintEntryBank.fingerprint(args.entry_bank) if args.entry_bank else None
            if fingerprints!=resume_checkpoint['config'].get('entry_bank_sha256'):
                raise ValueError('Resume cannot change the walking entry bank or its models')
            if args.entry!=resume_checkpoint['config'].get('entry','reset'):
                raise ValueError('Resume cannot change the walking entry distribution')
            controller_path=getattr(args,'walking_controller',None)
            controller_hash=hashlib.sha256(Path(controller_path).read_bytes()).hexdigest() if controller_path else None
            if controller_hash!=resume_checkpoint['config'].get('walking_controller_sha256'):
                raise ValueError('Resume cannot change the composed walking controller')
            if getattr(args,'walking_controller_ablation','composed')!=resume_checkpoint['config'].get('walking_controller_ablation','composed'):
                raise ValueError('Resume cannot change the walking controller ablation')
    task=TASKS[args.skill];session=read_session()
    horizon=float(getattr(args,'walking_episode_seconds',0.))
    if horizon:
        if task.name!='walking' or not math.isfinite(horizon) or not 10.<=horizon<=60.:
            raise ValueError('Walking episode length must be finite, between 10 and 60 seconds')
        task=replace(task,seconds=horizon)
    evaluate_skill=run_suite;protocol=PROTOCOL_VERSION
    evaluation_kwargs={} if args.eval_scene_robot is None else {'scene_robot':args.eval_scene_robot}
    if args.roller_contract:
        if evaluation_kwargs:raise ValueError('Native roller evaluation has its own scene contract')
        if task.name!='roller' or args.entry!='reset' or args.random_commands:raise ValueError('Native roller contract uses its own task starts and command cases')
        from .roller_evaluate import run_suite as evaluate_skill
        from .roller_tasks import PROTOCOL,CONDITIONS
        protocol=PROTOCOL
        if args.conditions is None:args.conditions=','.join(CONDITIONS)
    if args.scene_robot:
        if args.roll_starts:raise ValueError('Source-state curriculum requires its original articulated scene')
        task=replace(task,robot=args.scene_robot)
    from .budget import remaining,require_supervision
    require_supervision(SESSION)
    budget_reserve=float(getattr(args,'reserve_seconds',5400.)) if (SESSION/'active_budget.json').exists() else 60.
    if not math.isfinite(budget_reserve) or budget_reserve<0:raise ValueError('Invalid closeout reserve')
    start=time.time();job_end=time.monotonic()+args.minutes*60
    def time_left():return 0. if _STOP_REQUESTED else min(remaining(SESSION,reserve=budget_reserve),job_end-time.monotonic())
    deadline=start+time_left()  # Audit estimate only; enforcement uses the live ledger.
    if time_left()<=0:raise RuntimeError("Experiment budget or reserved closeout boundary reached")
    out=SESSION/"runs"/args.name;out.mkdir(parents=True,exist_ok=False)
    source=Path(args.source) if args.source else task.source
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(start_unix=start,deadline_unix=deadline,source=str(source),protocol=protocol)
    config["code_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    package=Path(__file__).parents[1]
    for name in ['obs.py','coords.py','godot_proc.py','policy_time.py','policy_memory.py','policy_state.py','policy_task_state.py','play_input.py','motion_control.py','backends/mujoco_backend.py','backends/godot_backend.py']:
        config['code_sha256']['sim2sim/'+name]=hashlib.sha256((package/name).read_bytes()).hexdigest()
    (out/"config.json").write_text(json.dumps(config,indent=2))
    template=Path(args.template) if args.template else source
    config["increment_template_sha256"]=hashlib.sha256(template.read_bytes()).hexdigest()
    time_gate=None if not args.time_gate else tuple(float(x) for x in args.time_gate.split(','))
    if args.command_gate and (task.name!='roller' or not args.roller_contract):
        raise ValueError('Negative-throttle gate requires the native roller command contract')
    policy=Policy(source,args.variant,args.std,args.bound,template=template,time_gate=time_gate,command_gate=args.command_gate,mask_task_state=getattr(args,'mask_task_state',False),action_basis=getattr(args,'action_basis',''),mask_motion_state=getattr(args,'mask_motion_state',False))
    policy.task_name=task.name
    policy.roller_contract=args.roller_contract
    retention=None
    if getattr(args,'teacher_mode',''):
        if task.name=='walking':
            from .walking_retention import WalkingTeacherRetention as TeacherRetention
        else:
            from .teacher_retention import TeacherRetention
        retention=TeacherRetention(policy,args.teacher_mode,args.teacher_replay,args.teacher_samples)
        config['teacher_retention']=retention.audit
        if resume_checkpoint is not None and retention.audit!=resume_checkpoint['config'].get('teacher_retention'):
            raise ValueError('Resume cannot change teacher data or its retention contract')
    critic=Critic(template,EXTRA_DIM,time_input_s=policy.anchor.time_input_s,heading_input=policy.anchor.heading_input,yaw_memory_input=policy.anchor.yaw_memory_input,state_input=policy.anchor.state_input,input_dim=policy.anchor.obs_dim)
    if policy.anchor.task_input and args.symmetry_weight:raise ValueError('Task-state symmetry is not defined')
    if policy.anchor.yaw_memory_input and args.symmetry_weight:raise ValueError('Walking memory needs its own reflection contract')
    obs_sign=observation_sign(task.name,policy.anchor.heading_input,policy.anchor.yaw_memory_input,policy.anchor.state_input)
    from .learner_device import LearnerDevice
    learner=LearnerDevice(policy,critic,getattr(args,'learner_device','cpu'))
    config['learner_device']=learner.device.type
    config['collection_device']='cpu'
    config['learner_actor_dtype']=str(next(policy.delta.net.parameters()).dtype)
    config['learner_critic_dtype']=str(next(critic.parameters()).dtype)
    if learner.device.type=='cuda':
        config['accelerator']=dict(name=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda)
    if retention is not None:retention.to(learner.device)
    actor_parameters=list(policy.delta.net.parameters())+[policy.log_std]
    ao=torch.optim.Adam(actor_parameters,lr=args.actor_lr);co=torch.optim.Adam(critic.parameters(),lr=args.critic_lr)
    initial_iteration=0
    if args.resume:
        ck=resume_checkpoint
        if ck["factory_sha256"]!=policy.anchor.sha256:raise RuntimeError("Resume source mismatch")
        policy.load_state_dict(ck["policy"]);critic.load_state_dict(ck["critic"])
        ao.load_state_dict(ck["actor_optimizer"]);co.load_state_dict(ck["critic_optimizer"])
        initial_iteration=ck["iteration"]
        random.setstate(ck["rng"]["python"]);np.random.set_state(ck["rng"]["numpy"]);torch.set_rng_state(ck["rng"]["torch"])
        if 'cuda' in ck['rng'] and learner.device.type=='cuda':torch.cuda.set_rng_state_all(ck['rng']['cuda'])
        config['resume_learner_device_from']=ck['config'].get('learner_device','cpu')
    learner.sync_collectors()
    references=reference_observations(task).to(learner.device) if args.variant=="anchor" else None
    initial_export=export_policy(policy,out/"initial.onnx")
    initial_parity=parity(policy,initial_export,n=1000)
    (out/"initial_parity.json").write_text(json.dumps(initial_parity,indent=2))
    if not initial_parity["passed"]:raise RuntimeError("Initial actor export parity failed")
    weights=json.loads(args.weights)
    training_conditions=None if not args.conditions else args.conditions.split(",")
    environment_state=None
    if resume_checkpoint is not None:
        environment_state=resume_checkpoint.get('environment')
        if environment_state is None:
            prior=resume_checkpoint['config']
            environment_state=dict(seed=prior.get('seed',args.seed),count=initial_iteration*prior.get('envs',16)*prior.get('steps',512)+prior.get('envs',16),rng=None)
            config['environment_resume']='legacy_disjoint_episode_seed_range'
        else:config['environment_resume']='restored_generator_and_episode_counter'
    motion_settings=None
    if getattr(args,'motion_config',None):
        if task.name!='walking' or args.walking_objective not in ('sprint_v1','sprint_v2'):raise ValueError('Feedback training requires the sprint objective')
        config['motion_config_sha256']=hashlib.sha256(Path(args.motion_config).read_bytes()).hexdigest()
        if resume_checkpoint is not None and config['motion_config_sha256']!=resume_checkpoint['config'].get('motion_config_sha256'):
            raise ValueError('Resume cannot change the motion feedback contract')
        motion_settings=json.loads(Path(args.motion_config).read_text())['walk']
        evaluation_kwargs['motion_settings']=motion_settings
    elif resume_checkpoint is not None and resume_checkpoint['config'].get('motion_config_sha256'):
        raise ValueError('Resume requires the original motion feedback contract')
    composition=None
    if getattr(args,'walking_controller',None):
        if task.name!='walking' or args.walking_objective!='sprint_v2' or not training_conditions or any(not c.startswith('sprint_') for c in training_conditions) or args.random_commands or motion_settings is None:
            raise ValueError('Composed walking requires explicit sprint_v2 feedback tapes')
        from .sprint_composition import SprintComposition
        composition=SprintComposition(args.walking_controller,learn_all=args.walking_controller_ablation=='learn_all')
        config['walking_controller_sha256']=composition.actor.sha256
        config['walking_training_contract']='native_composition_v1_'+args.walking_controller_ablation
        config['training_evaluator_scope']='single actor diagnostic; promotion requires separate native composite suite'
    env=Vector(task,args.envs,args.seed,weights,training_conditions,args.entry,args.roll_starts,json.loads(args.reward_params),args.random_commands,policy.anchor.time_input_s,policy.anchor.heading_input,environment_state,args.entry_source,args.entry_bank,args.roller_contract,policy.anchor.yaw_memory_input,policy.anchor.state_input,getattr(args,'roller_objective','legacy'),policy.anchor.task_input,getattr(args,'walking_objective','legacy'),motion_settings,composition)
    if env.entry_bank is not None:config['entry_bank_sha256']=env.entry_bank.hashes
    if args.entry_source:config['entry_source_sha256']=hashlib.sha256(Path(args.entry_source).read_bytes()).hexdigest()
    config["time_input_s"]=policy.anchor.time_input_s
    config["heading_input"]=policy.anchor.heading_input
    config['yaw_memory_input']=policy.anchor.yaw_memory_input
    config['state_input']=policy.anchor.state_input
    config['task_input']=policy.anchor.task_input
    config['environment_seed']=env.seed
    eval_entry=args.eval_entry or ("both" if args.entry=="mixed" else args.entry)
    eval_seeds=range(getattr(args,'eval_seed_start',100),getattr(args,'eval_seed_start',100)+getattr(args,'eval_seeds',3))
    config["physics"]=env.worlds[0].physics
    if env.roll_library is not None:config["roll_starts_sha256"]=env.roll_library.sha256
    if env.objectives[0].motion is not None:config["roll_motion_sha256"]=env.objectives[0].motion.sha256
    if any(w.physics!=config["physics"] for w in env.worlds):raise RuntimeError("Worker physics fingerprints differ")
    config["resume_starts_new_physical_episodes"]=bool(args.resume)
    (out/"config.json").write_text(json.dumps(config,indent=2))
    log=(out/"metrics.jsonl").open("a",buffering=1)
    best_score=-math.inf;best_success=-1.;iteration=initial_iteration;last_eval=time.time();total_samples=0;status="time_limit"
    evaluation_records=[]
    save_checkpoint(out/"latest.pt",policy,critic,ao,co,iteration,config,env)
    try:
        while time_left()>0 and (not args.iterations or iteration<initial_iteration+args.iterations):
            if (out/"STOP").exists():status="stopped_for_review";break
            iteration+=1;iteration_start=time.time()
            buffers={k:[] for k in ["obs","critic_obs","anchor","action","logprob","mean","std","value","reward","physical_reward","done","actor_mask"]}
            faults=0;all_terms={};term_count=0
            policy.train()
            collect_policy=learner.collect_policy;collect_critic=learner.collect_critic
            collection_start=time.perf_counter()
            with torch.no_grad():
                for step in range(args.steps):
                    if step%25==0 and time_left()<=0:break
                    obs,cobs=env.observations();anchor=collect_policy.anchor_values(obs)
                    dist=collect_policy.distribution(obs,anchor);action=dist.sample();value=collect_critic(cobs)
                    actor_mask=env.actor_mask()
                    reward,done,timeouts,terminal_cobs,terms=env.step(action.numpy())
                    if hasattr(env,'nonfinite_terminal'):
                        torch.save(env.nonfinite_terminal,out/'nonfinite_terminal.pt')
                        raise FloatingPointError('Nonfinite terminal critic observation; preserved nonfinite_terminal.pt')
                    physical_reward=reward.clone()
                    # Truncated time limits bootstrap terminal state, never the reset state.
                    reward=reward+args.gamma*collect_critic(terminal_cobs)*timeouts
                    for k,v in [("obs",obs),("critic_obs",cobs),("anchor",anchor),("action",action),("logprob",dist.log_prob(action).sum(-1)),("mean",dist.mean),("std",dist.stddev),("value",value),("reward",reward),("physical_reward",physical_reward),("done",done),("actor_mask",actor_mask)]:buffers[k].append(v)
                    for entry in terms:
                        for k,v in entry.items():all_terms[k]=all_terms.get(k,0.)+v
                        term_count+=1
                if not buffers["obs"]:break
                b={k:torch.stack(v) for k,v in buffers.items()}
                T,N=b["value"].shape;total_samples+=T*N
                advantage=torch.zeros_like(b["reward"]);gae=torch.zeros(N)
                next_value=collect_critic(env.observations()[1])
                for s in reversed(range(T)):
                    active=(~b["done"][s]).float()
                    delta=b["reward"][s]+args.gamma*next_value*active-b["value"][s]
                    gae=delta+args.gamma*args.lam*active*gae;advantage[s]=gae;next_value=b["value"][s]
                returns=advantage+b["value"]
                if composition is None:advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
                else:
                    selected_advantage=advantage[b['actor_mask']]
                    if selected_advantage.numel()>1:
                        advantage=(advantage-selected_advantage.mean())/(selected_advantage.std()+1e-8)
                    else:advantage=torch.zeros_like(advantage)
                flat={k:v.flatten(0,1) for k,v in b.items()};advantage=advantage.flatten();returns=returns.flatten()
                if args.symmetry_weight:
                    flat["mirrored_obs"]=flat["obs"][:,OBS_PERM]*torch.from_numpy(obs_sign)
                    flat["mirrored_anchor"]=policy.anchor_values(flat["mirrored_obs"])
            collection_seconds=time.perf_counter()-collection_start
            learner_start=time.perf_counter()
            flat=learner.batch(flat);advantage=advantage.to(learner.device);returns=returns.to(learner.device)
            before=copy.deepcopy(policy.state_dict());optbefore=copy.deepcopy(ao.state_dict())
            losses=[];kls=[];teacher_losses=[];stop_actor=False;update_count=0
            for epoch in range(args.epochs):
                for ix in torch.randperm(T*N).split(args.minibatch):
                    pred=critic(flat["critic_obs"][ix]);closs=(pred-returns[ix]).square().mean()
                    if not torch.isfinite(closs):
                        torch.save(dict(batch=b,returns=returns,indices=ix,prediction=pred,
                                        iteration=iteration,environment=env.checkpoint_state()),out/'nonfinite_batch.pt')
                        raise FloatingPointError("nonfinite critic loss; preserved nonfinite_batch.pt")
                    co.zero_grad();closs.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.);co.step()
                    if stop_actor or iteration-initial_iteration<=args.critic_warmup:continue
                    ix=ix.to(learner.device);ix=ix[flat['actor_mask'][ix]]
                    if not len(ix):continue
                    dist=policy.distribution(flat["obs"][ix],flat["anchor"][ix])
                    with torch.no_grad():
                        old=torch.distributions.Normal(flat["mean"][ix],flat["std"][ix])
                        kl=torch.distributions.kl_divergence(old,dist).sum(-1).mean()
                    kls.append(float(kl))
                    if float(kl)>2*args.target_kl:stop_actor=True;continue
                    logprob=dist.log_prob(flat["action"][ix]).sum(-1)
                    ratio=(logprob-flat["logprob"][ix]).clamp(-20,20).exp()
                    loss=-torch.minimum(ratio*advantage[ix],ratio.clamp(.8,1.2)*advantage[ix]).mean()
                    if references is not None:
                        ref=references[torch.randint(len(references),(min(args.minibatch,len(references)),))]
                        loss=loss+args.anchor_weight*policy.delta(ref).square().mean()
                    if args.variant=="residual":loss=loss+args.residual_weight*policy.delta(flat["obs"][ix]).square().mean()
                    if retention is not None:
                        teacher_loss=retention.loss(policy,flat['obs'])
                        teacher_losses.append(float(teacher_loss.detach()))
                        loss=loss+args.teacher_weight*teacher_loss
                    if args.symmetry_weight:
                        mirrored=policy(flat["mirrored_obs"][ix],flat["mirrored_anchor"][ix])
                        reflected=dist.mean[:,JOINT_PERM]*torch.from_numpy(JOINT_SIGN).to(learner.device)
                        loss=loss+args.symmetry_weight*(mirrored-reflected).square().mean()
                    if not torch.isfinite(loss):raise FloatingPointError("nonfinite actor loss")
                    ao.zero_grad();loss.backward()
                    if iteration<=args.freeze_std:policy.log_std.grad=None
                    torch.nn.utils.clip_grad_norm_(actor_parameters,.5);ao.step()
                    with torch.no_grad():policy.log_std.clamp_(np.log(.005),np.log(.3))
                    losses.append(float(loss.detach()));update_count+=1
            with torch.no_grad():
                ix=torch.arange(0,T*N,max(1,T*N//2048))
                prediction=critic(flat["critic_obs"][ix]);truth=returns[ix]
                value_loss=float((prediction-truth).square().mean())
                explained_variance=float(1-(truth-prediction).var()/(truth.var()+1e-8))
                ix=ix.to(learner.device);ix=ix[flat['actor_mask'][ix]]
                dist=policy.distribution(flat["obs"][ix],flat["anchor"][ix])
                old=torch.distributions.Normal(flat["mean"][ix],flat["std"][ix])
                actual_kl=float(torch.distributions.kl_divergence(old,dist).sum(-1).mean()) if len(ix) else 0.
                delta_rms=float(policy.delta(flat["obs"][ix]).square().mean().sqrt()) if len(ix) else 0.
            rejected=not math.isfinite(actual_kl) or actual_kl>.15
            if rejected:
                policy.load_state_dict(before);ao.load_state_dict(optbefore)
                for g in ao.param_groups:g["lr"]*=.5
            elif actual_kl>args.target_kl*1.5:
                for g in ao.param_groups:g["lr"]=max(1e-6,g["lr"]*.8)
            elif getattr(args,'adaptive_lr_max',0.)>0 and update_count>0 and 0<actual_kl<args.target_kl/2:
                for g in ao.param_groups:
                    if g['lr']<args.adaptive_lr_max:g['lr']=min(args.adaptive_lr_max,g['lr']*1.2)
            learner.synchronize()
            learner_seconds=time.perf_counter()-learner_start
            sync_start=time.perf_counter();learner.sync_collectors();sync_seconds=time.perf_counter()-sync_start
            entry={"iteration":iteration,"elapsed":time.time()-start,"samples":total_samples,"reward":float(b["physical_reward"].mean()),"bootstrapped_reward":float(b["reward"].mean()),"reward_logging":"physical_v2","value_loss":value_loss,"explained_variance":explained_variance,"delta_rms":delta_rms,"kl":actual_kl,"actor_lr":ao.param_groups[0]["lr"],"std":float(policy.log_std.detach().exp().mean()),"rejected":rejected,"actor_updates":update_count,"fps":T*N/(time.time()-iteration_start),"done_fraction":float(b["done"].float().mean()),"terms":{k:v/term_count for k,v in all_terms.items()}}
            entry.update(learner_device=learner.device.type,collection_seconds=collection_seconds,
                         learner_seconds=learner_seconds,collector_sync_seconds=sync_seconds)
            if composition is not None:entry.update(actor_samples=int(b['actor_mask'].sum()),controller_steps=dict(composition.counts))
            if retention is not None:entry['teacher_kl']=float(np.mean(teacher_losses)) if teacher_losses else 0.
            log.write(json.dumps(entry)+"\n")
            print(json.dumps({k:v for k,v in entry.items() if k!="terms"}),flush=True)
            # Only atomic, completed updates are eligible for automatic resume.
            save_checkpoint(out/"latest.pt",policy,critic,ao,co,iteration,config,env)
            evaluate_now=time.time()-last_eval>args.eval_seconds or (args.iterations and iteration>=initial_iteration+args.iterations)
            if evaluate_now and not _STOP_REQUESTED and remaining(SESSION,reserve=budget_reserve+30)>0:
                export=export_policy(policy,out/f"iteration_{iteration:05d}.onnx")
                save_checkpoint(out/f"iteration_{iteration:05d}.pt",policy,critic,ao,co,iteration,config,env)
                # Report export roundoff independently of physical outcomes.
                report=parity(policy,export,n=200)
                selected=None if not args.eval_conditions else args.eval_conditions.split(",")
                result=evaluate_skill(task.name,export,seeds=eval_seeds,workers=4,out=out/f"eval_{iteration:05d}",selected_conditions=selected,entry=eval_entry,**evaluation_kwargs)
                result_short={k:v for k,v in result.items() if k!="episodes"};result_short.update(iteration=iteration,parity=report)
                evaluation_records.append(result_short)
                (out/"evaluations.json").write_text(json.dumps(evaluation_records,indent=2))
                print("EVALUATION "+json.dumps(result_short),flush=True)
                if report["passed"] and not result["errors"] and (result["success_rate"],result["score"])>(best_success,best_score):
                    best_score=result["score"]
                    best_success=result["success_rate"]
                    save_checkpoint(out/"best.pt",policy,critic,ao,co,iteration,config,env)
                    (out/"best.onnx").write_bytes(export.read_bytes())
                last_eval=time.time()
            if rejected and ao.param_groups[0]["lr"]<1e-6:status="repeated_kl_rejection";break
        save_checkpoint(out/"latest.pt",policy,critic,ao,co,iteration,config,env)
        if args.iterations and iteration>=initial_iteration+args.iterations and status=='time_limit':
            status='iteration_limit'
        if _STOP_REQUESTED:status="interrupted_checkpointed"
        export=export_policy(policy,out/"final.onnx")
        report=parity(policy,export,n=1000)
        (out/"final_parity.json").write_text(json.dumps(report,indent=2))
        if not _STOP_REQUESTED and remaining(SESSION,reserve=budget_reserve+30)>0:
            selected=None if not args.eval_conditions else args.eval_conditions.split(",")
            result=evaluate_skill(task.name,export,seeds=eval_seeds,workers=4,out=out/"eval_final",selected_conditions=selected,entry=eval_entry,**evaluation_kwargs)
            if report["passed"] and not result["errors"] and (result["success_rate"],result["score"])>(best_success,best_score):
                (out/"best.onnx").write_bytes(export.read_bytes());save_checkpoint(out/"best.pt",policy,critic,ao,co,iteration,config,env);best_score=result["score"];best_success=result["success_rate"]
        (out/"completed.json").write_text(json.dumps({"status":status,"iterations":iteration,"samples":total_samples,"elapsed":time.time()-start,"best_dev_score":best_score if math.isfinite(best_score) else None,"best_dev_success":best_success,"final_parity":report,"entry_counts":env.entry_counts},indent=2))
    except BaseException as e:
        (out/"error.txt").write_text(traceback.format_exc())
        save_checkpoint(out/"failed.pt",policy,critic,ao,co,iteration,config,env)
        raise
    finally:env.close();log.close()
    return out

def main():
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,request_stop)
    p=argparse.ArgumentParser()
    p.add_argument("--skill",required=True,choices=list(TASKS));p.add_argument("--name",required=True)
    p.add_argument("--variant",choices=["plain","anchor","residual"],default="residual")
    p.add_argument('--walking-episode-seconds',type=float,default=0.,help='Explicit walking training horizon; zero preserves the task default and does not change native evaluation programs')
    p.add_argument("--minutes",type=float,default=15);p.add_argument("--iterations",type=int,default=0)
    p.add_argument('--reserve-seconds',type=float,default=5400.,help='Session closeout reserve; old eight-hour sessions retain 90 minutes by default')
    p.add_argument('--walking-objective',choices=['legacy','sprint_v1','sprint_v2'],default='legacy')
    p.add_argument('--motion-config',type=Path,help='Explicit game command feedback shared during sprint sampling and evaluation')
    p.add_argument('--walking-controller',type=Path,help='Train only sprint actions; execute this frozen ordinary actor on all other tape steps')
    p.add_argument('--walking-controller-ablation',choices=['composed','learn_all'],default='composed',help='Matched control: learn_all keeps the same phase observations/feedback but also learns ordinary actions')
    p.add_argument("--envs",type=int,default=16);p.add_argument("--steps",type=int,default=512)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--threads",type=int,default=2)
    p.add_argument('--learner-device',choices=['auto','cpu','cuda'],default='auto',help='GPU batch updates when CUDA is available; Jolt collection and frozen ORT anchor stay on CPU')
    p.add_argument("--epochs",type=int,default=2);p.add_argument("--minibatch",type=int,default=2048)
    p.add_argument("--actor-lr",type=float,default=3e-5);p.add_argument("--critic-lr",type=float,default=3e-4)
    p.add_argument('--adaptive-lr-max',type=float,default=0.,help='Optional upward KL adaptation cap; zero preserves the prior conservative schedule')
    p.add_argument("--std",type=float,default=.03);p.add_argument("--bound",type=float,default=.2)
    p.add_argument("--freeze-std",type=int,default=20);p.add_argument("--critic-warmup",type=int,default=2)
    p.add_argument("--gamma",type=float,default=.99);p.add_argument("--lam",type=float,default=.95)
    p.add_argument("--target-kl",type=float,default=.015);p.add_argument("--anchor-weight",type=float,default=10)
    p.add_argument("--residual-weight",type=float,default=1);p.add_argument("--weights",default="{}")
    p.add_argument("--conditions");p.add_argument("--eval-conditions");p.add_argument("--eval-seconds",type=float,default=300)
    p.add_argument("--eval-seed-start",type=int,default=100);p.add_argument("--eval-seeds",type=int,default=3)
    p.add_argument("--source");p.add_argument("--resume");p.add_argument("--template")
    p.add_argument("--entry",choices=["reset","standing","mixed"],default="reset")
    p.add_argument("--eval-entry",choices=["reset","standing","both"])
    p.add_argument("--entry-source",help="Optional deployed idle actor used for training handoffs; standard evaluation keeps its original entry actor")
    p.add_argument("--entry-bank",help="Training-only native controller prefixes from a declared deployment bank")
    p.add_argument("--scene-robot",choices=['microduck','microduck_ball','microduck_ball_stand_fix','microduck_roller'])
    p.add_argument("--eval-scene-robot",choices=['microduck','microduck_ball','microduck_ball_stand_fix'],help='Explicit separate evaluation scene; omitted keeps the original task scene')
    p.add_argument("--roller-contract",choices=['native'],help='Use native push/coast/brake and relative-heading tasks for roller')
    p.add_argument('--roller-objective',choices=['legacy','command_heading_v1','stop_hold_v1'],default='legacy',help='Explicit objective version; legacy retains sealed experiment semantics')
    p.add_argument('--mask-task-state',action='store_true',help='68D actor ablation: keep architecture and critic fixed but hide the seven added actor features')
    p.add_argument('--mask-motion-state',action='store_true',help='Declared state actor ablation: hide planar velocity and height from the residual, retaining the same critic')
    p.add_argument('--action-basis',choices=['','brake_sagittal_v1'],default='',help='Restrict brake learning and exploration to hip pitch and knee targets; preserve the anchor elsewhere')
    p.add_argument('--teacher-mode',choices=['','online_kl','replay_kl'],default='',help='Compare phase-balanced teacher KL on current versus successful teacher occupancy')
    p.add_argument('--teacher-replay',help='Completed, checksummed teacher-data manifest; only for replay_kl')
    p.add_argument('--teacher-weight',type=float,default=.02)
    p.add_argument('--teacher-samples',type=int,default=384)
    p.add_argument("--roll-starts",type=float,default=0.,help="Training-only fraction of source mid-roll resets")
    p.add_argument("--symmetry-weight",type=float,default=0.,help="Bilateral actor consistency loss using the upstream observation/action transform")
    p.add_argument("--reward-params",default="{}",help="Explicit tracking-kernel variances")
    p.add_argument("--random-commands",type=float,default=0.,help="Fraction of training episodes with varied interactive command tapes")
    p.add_argument("--time-gate",default="",help="Optional start,end seconds for a learned increment on a declared time-input actor")
    p.add_argument('--command-gate',choices=['','negative_throttle'],default='',help='Only adapt negative roller throttle; preserve factory push/coast exactly')
    args=p.parse_args()
    if args.walking_controller_ablation!='composed' and not args.walking_controller:p.error('--walking-controller-ablation requires --walking-controller')
    if args.teacher_replay and args.teacher_mode!='replay_kl':p.error('--teacher-replay requires --teacher-mode replay_kl')
    if args.teacher_mode and (not math.isfinite(args.teacher_weight) or args.teacher_weight<=0):p.error('--teacher-weight must be positive and finite')
    if not 0<=args.roll_starts<=1:p.error("--roll-starts must be in [0,1]")
    if not 0<=args.random_commands<=1:p.error("--random-commands must be in [0,1]")
    out=SESSION/"runs"/args.name;existed_before=out.exists()
    try:print(run(args),flush=True)
    except BaseException:
        if not existed_before and out.is_dir() and not (out/"error.txt").exists():
            (out/"error.txt").write_text(traceback.format_exc())
        raise

if __name__=="__main__":main()
