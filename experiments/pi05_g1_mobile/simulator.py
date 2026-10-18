"""T2 release PhysX runner. New task/acceptance must be frozen before scoring.

Run in the owned source image via its Isaac interpreter, never host system Python.
CLI: python -m experiments.pi05_g1_mobile.simulator --config CONFIG.json
Diagnostic mode is reset plus at most ten self-state hold controls, zero VLA.
Only policy RGB/measured joints/state/text cross IPC. Object truth stays in logs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from experiments.pi05_g1_round02.simulator import (
    PolicyProcess, canonical_config, digest, refresh_camera, intersect_action_bounds)

OBJECTS = ('brown_box', 'blue_sorting_bin')
SOURCE_BUILD = '6.0.0-rc.22+release.33481.407f3ea1.gl'
VIDEO_OUTPUT_PARAMS = ['-movflags', 'frag_keyframe+empty_moov+default_base_moof', '-g', '50']


class InfraWallDeadlineError(RuntimeError):
    """Infrastructure deadline; never a measured task failure/completion."""


def check_wall_deadline(config, result, now_ns=None):
    limit = config.get('max_run_wall_seconds')
    if limit is not None:
        elapsed = ((time.monotonic_ns() if now_ns is None else now_ns) -
                   result['cold_process_started_monotonic_ns']) / 1e9
        if elapsed >= limit:
            raise InfraWallDeadlineError('infra_wall_deadline: elapsed=' + str(elapsed) + ' limit=' + str(limit))


def close_recordings_before_model(video, observer_video, model):
    errors = []
    for item in (video, observer_video, model):
        if item is not None:
            try:item.close()
            except Exception as exc:errors.append(repr(exc))
    return errors


def runtime_identity_payload(config, result, identity):
    """Static prefix-audit metadata only; never added to model RPC inputs."""
    return {'schema':'pi05_g1_mobile_runtime_identity_v1','sdk_build':SOURCE_BUILD,
            'source_joint_names':result['source_joint_names'],'body_names':result['body_names'],
            'contact_view':result['contact_view'],'wbc_config':result['wbc_config'],
            'frozen_interface_sha256':result['frozen_interface_sha256'],
            'interface':identity,'frozen_files':config.get('frozen_files',{}),
            'offline_unused_assets':result.get('offline_unused_assets'),
            'video_encoding':{'fps':50,'codec':'libx264','quality':'unchanged imageio default',
                'output_params':list(VIDEO_OUTPUT_PARAMS),
                'bytes_changed_by_fragmentation_and_GOP50':True,
                'raw_policy_RGB_unchanged_by_video_encoder':True},
            'scope':'Static interruption/prefix audit identity; failed or truncated episode is never formal evidence'}


def validate_config(config):
    config = copy.deepcopy(config)
    if type(config.get('offline_unused_assets', False)) is not bool:
        raise ValueError('offline_unused_assets must be explicit boolean')
    deadline = config.get('max_run_wall_seconds')
    if deadline is not None and (type(deadline) not in (int,float) or not math.isfinite(deadline) or deadline<=0):
        raise ValueError('max_run_wall_seconds must be finite positive or None')
    if config.get('mode') == 'diagnostic_hold':config['mode']='diagnostic'
    if config.get('mode') not in ('diagnostic', 'policy', 'replay'):
        raise ValueError('mode must explicitly be diagnostic_hold/policy/replay')
    case = config['case']
    if not isinstance(case.get('case_id'), str) or not case['case_id'] or type(case.get('seed')) is not int:
        raise ValueError('Explicit case identity/seed required')
    episode = config.get('episode_seconds', 30.)
    if type(episode) not in (int, float) or not math.isfinite(episode) or episode <= 0:
        raise ValueError('Finite positive episode_seconds required')
    config['episode_seconds'] = float(episode)
    ticks = config.get('ticks', int(round(episode*50)))
    if type(ticks) is not int or ticks < 0 or ticks > round(episode*50):
        raise ValueError('ticks must fit explicit episode duration')
    if config['mode'] == 'diagnostic' and ticks > 10:
        raise ValueError('Diagnostic is bounded reset/0..10 controls only')
    config['ticks'] = ticks
    if config['mode'] == 'policy':
        if config.get('native_action_horizon', 50) != 50:
            raise ValueError('Mobile native horizon must be fifty')
        cadence = config.get('replan_interval_ticks')
        if type(cadence) is not int or not 1 <= cadence <= 50:
            raise ValueError('Explicit cadence 1..50 required')
        if not config.get('action_limits_file'):
            raise ValueError('Policy requires explicit root-approved action limits')
    for key in ('box_pose', 'bin_pose', 'robot_pose'):
        pose = config.get('scene', {}).get(key)
        if pose is not None:
            if set(pose) != {'position_xyz', 'rotation_xyzw'}:
                raise ValueError('Reset pose must contain position_xyz/rotation_xyzw only')
            p, q = pose['position_xyz'], pose['rotation_xyzw']
            if len(p) != 3 or len(q) != 4 or not all(math.isfinite(float(x)) for x in [*p,*q]):
                raise ValueError('Invalid reset pose')
            if abs(sum(float(x)**2 for x in q)-1.) > 1e-5:
                raise ValueError('Reset quaternion must be normalized SDK xyzw')
    return config


def apply_initial_scene(description, scene, pose_class):
    """Only pre-build initial configuration; never a runtime actor pose writer."""
    for key, name in (('box_pose','brown_box'),('bin_pose','blue_sorting_bin')):
        if key in scene:
            value = scene[key]
            description.scene.assets[name].set_initial_pose(pose_class(
                position_xyz=tuple(value['position_xyz']), rotation_xyzw=tuple(value['rotation_xyzw'])))
    if 'robot_pose' in scene:
        value = scene['robot_pose']
        description.embodiment.set_initial_pose(pose_class(
            position_xyz=tuple(value['position_xyz']), rotation_xyzw=tuple(value['rotation_xyzw'])))


def suppress_term(raw, name, events):
    cfg = copy.deepcopy(raw.termination_manager.get_term_cfg(name))
    original = cfg.func
    def audited(*args, **kwargs):
        value = original(*args, **kwargs)
        active = bool(value.any())
        record = events.setdefault(name, {'ever':False,'first_control_tick':None,'calls':0})
        record.update(value=active, calls=record['calls']+1)
        if active and not record['ever']:
            record.update(ever=True, first_control_tick=int(raw._sim_step_counter)//4)
        return value & False
    cfg.func = audited
    raw.termination_manager.set_term_cfg(name, cfg)


def verified_success_check(rows, config, evaluator):
    """External acceptance may end an episode; it never supplies policy actions.

    Cheap truth-only prefilter limits full audit cost. The independent evaluator
    remains the sole success judge; its metadata is never a model observation.
    """
    import numpy as np
    if (config.get('stop_on_verified_success') is not True or config.get('mode')!='policy'
            or config.get('evaluate_acceptance') is not True or not rows):
        return None
    row=rows[-1];tick=row['control_tick']
    if tick<100 or tick%50 or not row['original_terms'].get('success',{}).get('ever'):
        return None
    c=row['acceptance_contacts_only'];bf=np.asarray(c['box_bin_force_world_n'])
    rf=np.asarray(c['box_robot_force_matrix_world_n'])
    velocity=np.asarray(row['acceptance_truth_only']['brown_box']['velocity'])
    if (bf.shape!=(3,) or rf.shape!=(53,3) or velocity.shape!=(6,)
            or not all(np.isfinite(v).all() for v in (bf,rf,velocity))):
        return None
    if not (bf[2]>0 and not np.any(rf) and np.linalg.norm(velocity[:3])<.02
            and np.linalg.norm(velocity[3:])<.1):
        return None
    started=time.monotonic_ns()
    audit=evaluator(rows,config)
    return {'control_tick':tick,'success':audit.get('success') is True,
            'audit_wall_seconds':(time.monotonic_ns()-started)/1e9}


def make_contact_view(raw):
    """Existing reports only; all robot bodies and actual bin rigid prims."""
    from pxr import UsdPhysics, PhysxSchema
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    box = '/World/envs/env_0/brown_box/Geometry/brown_box'
    if not stage.GetPrimAtPath(box).HasAPI(PhysxSchema.PhysxContactReportAPI):
        raise ValueError('Original brown-box contact report missing; no schema added')
    bin_paths = [str(p.GetPath()) for p in stage.Traverse()
                 if str(p.GetPath()).startswith('/World/envs/env_0/blue_sorting_bin') and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not bin_paths:
        raise ValueError('Actual bin rigid-body paths unavailable')
    names = list(raw.scene['robot'].body_names)
    if len(names)!=53:raise ValueError('Original53 robot bodies required')
    filters = bin_paths + ['/World/envs/env_0/Robot/'+name for name in names]
    view = raw.sim.physics_manager.get_physics_sim_view().create_rigid_contact_view(
        box, filter_patterns=filters, max_contact_data_count=4096)
    if view.sensor_count != 1 or view.filter_count != len(filters):
        raise ValueError('Box/robot/bin contact coverage mismatch')
    return view, {'sensor_paths':list(view.sensor_paths),'filter_paths':view.filter_paths,
                  'robot_body_names':names,'bin_rigid_paths':bin_paths,'schema_writes':0,
                  'bin_filter_indices':list(range(len(bin_paths))),
                  'robot_filter_indices':list(range(len(bin_paths),len(filters))),
                  'hand_filter_indices':[len(bin_paths)+i for i,n in enumerate(names) if 'hand_' in n],
                  'scope':'last200Hz solve; raw forces/point normals, acceptance only'}


def sample_contacts(view, tensor, metadata):
    import numpy as np
    def array(v):
        return tensor(v).detach().cpu().numpy()
    net = array(view.get_net_contact_forces(.005))
    matrix = array(view.get_contact_force_matrix(.005))
    force, point, normal, separation, count, start = map(array, view.get_contact_data(.005))
    points = []
    for index in range(view.filter_count):
        size, offset = int(count[0,index]), int(start[0,index])
        if size < 0 or offset < 0 or offset+size > view.max_contact_data_count:
            raise ValueError('Contact capacity exceeded')
        for i in range(offset, offset+size):
            points.append({'filter_index':index,'normal_force_n':float(force[i,0]),
                'position_world':point[i].tolist(),'normal_world':normal[i].tolist(),
                'separation_m':float(separation[i,0])})
    if not np.isfinite(net).all() or not np.isfinite(matrix).all():
        raise ValueError('Nonfinite contact telemetry')
    if matrix.shape!=(1,view.filter_count,3) or len(metadata['hand_filter_indices'])!=16:
        raise ValueError('Box/bin/53robot/16hand actual contact shape mismatch')
    return {'box_net_force_world_n':net.tolist(),'box_filter_force_matrix_world_n':matrix.tolist(),
            'box_hand_force_matrix_world_n':matrix[0,metadata['hand_filter_indices']].tolist(),
            'box_robot_force_matrix_world_n':matrix[0,metadata['robot_filter_indices']].tolist(),
            'box_bin_force_world_n':matrix[0,metadata['bin_filter_indices']].sum(axis=0).tolist(),
            'points':points}



class PreflightSceneTelemetry:
    """Read-only dynamic background + USD/camera pose diagnostics, audit only."""
    def __init__(self, raw):
        import omni.usd
        from pxr import UsdPhysics
        self.stage=omni.usd.get_context().get_stage()
        suffixes={'table':'/TaskAssets/table/Geometry/sm_tabletop_a01_01',
                  'drill':'/TaskAssets/power_drill_physics'}
        self.bodies={}
        for name,suffix in suffixes.items():
            paths=[str(p.GetPath()) for p in self.stage.Traverse()
                   if str(p.GetPath()).startswith('/World/envs/env_0/galileo_locomanip/')
                   and str(p.GetPath()).endswith(suffix) and p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(paths)!=1:raise ValueError('Preflight actual background owner ambiguous/missing: '+name)
            view=raw.sim.physics_manager.get_physics_sim_view().create_rigid_body_view(paths[0])
            if view.count!=1 or list(view.prim_paths)!=paths:
                raise ValueError('Preflight rigid-body view identity mismatch')
            self.bodies[name]=(paths[0],view)

    def sample(self, measured, observer):
        from pxr import Usd,UsdGeom
        cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'],useExtentsHint=True)
        bodies={}
        for name,(path,view) in self.bodies.items():
            # USD bounds can lag live PhysX when Fabric bypasses USD writes.
            # Keep them separately labelled from actual PhysX world transforms.
            bounds=cache.ComputeWorldBound(self.stage.GetPrimAtPath(path)).ComputeAlignedRange()
            bodies[name]={'prim_path':path,'physx_world_pose_xyzw':measured(view.get_transforms()).tolist(),
                'physx_world_velocity':measured(view.get_velocities()).tolist(),
                'usd_stage_world_bbox': [list(bounds.GetMin()),list(bounds.GetMax())],
                'usd_bbox_is_not_claimed_as_current_physx_shape_bounds':True}
        return {'background_dynamic_bodies':bodies,
                'observer_camera':{'pos_w':measured(observer.data.pos_w).tolist(),
                    'quat_w_ros_xyzw':measured(observer.data.quat_w_ros).tolist(),
                    'quat_w_world_xyzw':measured(observer.data.quat_w_world).tolist(),
                    'quat_w_opengl_xyzw':measured(observer.data.quat_w_opengl).tolist()},
                'uses':'preflight/audit only; never policy input or action decision'}


def run(input_config):
    import numpy as np
    from experiments.pi05_g1_mobile.contract import encode_state, decode_to_source, UPPER_NAMES
    config = validate_config(input_config)
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
    case, assets = config['case'], config['assets']
    result = {'schema':'pi05_g1_mobile_case_result_v1','status':'running','success':None,
              'case_id':case['case_id'],'case':case,'policy':config.get('policy_id',config['mode']),
              'control_hz':50,'physics_hz':200,'simulation_pauses_for_inference':True,
              'real_time_1x_verified':False,'cold_process_started_monotonic_ns':time.monotonic_ns(),
              'source_interface_passed':False,'inference_calls':[],
              'external_success_audits':[],
              'success_termination_semantics':'Optional external physics audit ends episode; not VLA autonomous recognition of done; never writes actions',
              'head_video_path':str(output/'head_50fps.mp4'),'observer_video_path':str(output/'observer_50fps.mp4'),
              'HFmodel':config.get('HFmodel'),'scene_reset_overrides':config.get('scene',{})}
    model = video = observer_video = env = None
    rows = []
    def finish(error=None):
        nonlocal model,video,observer_video
        if error is not None:
            result.update(status='failed',success=None,error=repr(error))
            if isinstance(error,InfraWallDeadlineError):
                result.update(termination_reason='infra_wall_deadline',infrastructure_failure=True)
        errors=close_recordings_before_model(video,observer_video,model)
        if errors:
            result.setdefault('cleanup_errors',[]).extend(errors)
            result.update(status='failed',success=None)
        model = video = observer_video = None
        if (output/'trajectory.jsonl').exists():result['trace_sha256']=digest(output/'trajectory.jsonl')
        result['evidence_finalized_monotonic_ns']=time.monotonic_ns()
        (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    # Install before any Arena import; no selected asset or physical source edits.
    if config.get('offline_unused_assets', False):
        try:
            from experiments.pi05_g1_mobile.offline_assets import install_offline_assets
            result['offline_unused_assets'] = install_offline_assets()
        except Exception as exc:
            finish(exc)
            raise
    # Source imports before Kit must follow the official SimulationAppContext.
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
    from isaaclab_arena_environments.galileo_g1_locomanip_pick_and_place_environment import GalileoG1LocomanipPickAndPlaceEnvironment
    parser = get_isaaclab_arena_cli_parser()
    GalileoG1LocomanipPickAndPlaceEnvironment.add_cli_args(parser)
    launch = parser.parse_args(['--headless','--enable_cameras','--no-solve-relations','--embodiment','g1_wbc_joint','--object','brown_box'])
    launch.seed=case['seed'];sys.argv=[sys.argv[0]]
    class EvidenceContext(SimulationAppContext):
        def __exit__(self,typ,value,tb):
            finish(value)
            return super().__exit__(typ,value,tb)
    try:
        with EvidenceContext(launch):
            import torch
            import warp as wp
            import imageio.v2 as imageio
            from PIL import Image
            # Same frozen GB10/NVRTC compatibility switch as the proven source
            # runner. Disables Torch's legacy CUDA JIT fuser, not PhysX/control.
            torch._C._jit_override_can_fuse_on_gpu(False)
            from isaaclab_arena.assets.background_library import GalileoLocomanipBackground
            from isaaclab_arena.assets.object_library import BrownBox,BlueSortingBin
            from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
            from isaaclab_arena.utils.pose import Pose
            import isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_joint_action as action_module
            import isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.g1 as robot_module
            if Path('/isaac-sim/VERSION').read_text().strip()!=SOURCE_BUILD:
                raise ValueError('T2 source SDK build mismatch')
            for path,expected in config.get('frozen_files',{}).items():
                if digest(path)!=expected:raise ValueError('Frozen input changed: '+path)
            def tensor(v):return wp.to_torch(v) if isinstance(v,wp.array) else v
            def measured(v):return tensor(v)[0].detach().cpu().numpy()
            factory=action_module.get_wbc_policy
            def local_factory(name,robot,cfg,count):
                if cfg.wbc_version!='homie_v2' or cfg.enable_waist is not False:
                    raise ValueError('Original HomieV2 with disabled waist required')
                cfg.wbc_model_path=','.join(str(assets[k]) for k in ('stand','walk'))
                result['wbc_config']=canonical_config(vars(cfg))
                return factory(name,robot,cfg,count)
            action_module.get_wbc_policy=local_factory
            def local_robot(url,**kwargs):
                if url.endswith('/g1_29dof_with_hand.urdf'):return str(assets['urdf'])
                if url.endswith('/Arena/wbc_policy/robot_model/g1/'):return str(Path(assets['urdf']).parent)
                mapping=config.get('mesh_url_paths',{})
                if url in mapping:return mapping[url]
                raise ValueError('Unexpected uncached robot asset: '+url)
            robot_module.retrieve_file_path=local_robot
            GalileoLocomanipBackground.usd_path=str(assets['background'])
            BrownBox.usd_path=str(assets['box']);BlueSortingBin.usd_path=str(assets['bin'])
            description=GalileoG1LocomanipPickAndPlaceEnvironment().get_env(launch)
            apply_initial_scene(description,config.get('scene',{}),Pose)
            description.task.episode_length_s=config['episode_seconds']
            description.embodiment.scene_config.robot.spawn.usd_path=str(assets['usd'])
            from isaaclab.sensors import CameraCfg
            import isaaclab.sim as sim_utils
            from isaaclab.utils import configclass
            from isaaclab_arena.utils.configclass import combine_configclass_instances
            observer_cfg=config['observer_camera']
            @configclass
            class ObserverScene:
                mobile_observer:CameraCfg=CameraCfg(prim_path='{ENV_REGEX_NS}/MobileObserver',update_period=0.,
                    height=observer_cfg['height'],width=observer_cfg['width'],data_types=['rgb'],
                    spawn=sim_utils.PinholeCameraCfg(focal_length=observer_cfg['focal_length'],clipping_range=tuple(observer_cfg['clipping_range'])),
                    offset=CameraCfg.OffsetCfg(pos=tuple(observer_cfg['position']),rot=tuple(observer_cfg['rotation_xyzw']),convention='ros'))
            description.embodiment.scene_config=combine_configclass_instances('MobileScene',description.embodiment.scene_config,ObserverScene())
            env=ArenaEnvBuilder(description,launch).make_registered();raw=env.unwrapped
            if raw.cfg.sim.dt!=.005 or raw.cfg.decimation!=4 or raw.cfg.episode_length_s!=config['episode_seconds']:
                raise ValueError('Source dt/controller period/episode mismatch')
            if tuple(env.action_space.shape)!=(1,50):raise ValueError('Source action width changed')
            robot,camera,observer=raw.scene['robot'],raw.scene['robot_head_cam'],raw.scene['mobile_observer']
            term=raw.action_manager.get_term('g1_action');names=list(robot.joint_names)
            upper_ids=[names.index(n) for n in UPPER_NAMES];rest_ids=[i for i in range(43) if i not in upper_ids]
            if len(upper_ids)!=28 or len(rest_ids)!=15:raise ValueError('Effective upper28/rest15 coverage mismatch')
            if len(term.robot_model.get_joint_group_indices('upper_body'))!=28:
                raise ValueError('Original WBC upper group is not effective28')
            env.reset(seed=case['seed'])
            events={}
            for name in ('success','time_out','object_dropped'):suppress_term(raw,name,events)
            result['source_terminal_autoresets_suppressed']=['success','time_out','object_dropped']
            refresh_camera(raw,robot,camera,OBJECTS,measured)
            observer.reset();_=observer.data.output['rgb']
            contacts,contact_meta=make_contact_view(raw)
            preflight_telemetry=PreflightSceneTelemetry(raw) if config.get('capture_preflight_scene_state') is True else None
            result.update(source_joint_names=names,body_names=list(robot.body_names),contact_view=contact_meta,
                          source_interface_passed=True,source_waist_enabled=False)
            audit_context={**config,'audit_body_names':list(robot.body_names),'audit_contact_metadata':contact_meta}
            identity={'scene':canonical_config(raw.cfg.scene.to_dict()),'physics':canonical_config(raw.cfg.sim.to_dict()),
                'head_camera':canonical_config(camera.cfg.to_dict()),'observer':canonical_config(observer.cfg.to_dict()),
                'action':canonical_config(term.cfg.to_dict()),'scene_config':config.get('scene',{}),
                'replan_interval_ticks':config.get('replan_interval_ticks'),
                'episode_seconds':config['episode_seconds'],'simulator_sha256':digest(__file__),
                'contract_sha256':digest(Path(__file__).with_name('contract.py')),'assets':{k:digest(v) for k,v in assets.items()}}
            limits=None
            if config.get('action_limits_file'):
                bounds=json.loads(Path(config['action_limits_file']).read_text())
                limits=intersect_action_bounds(bounds['lower'],bounds['upper'],measured(robot.data.soft_joint_pos_limits),names)
                result['actual_action_bounds']={'lower':limits[0].tolist(),'upper':limits[1].tolist()}
                result['action_limits_sha256']=digest(config['action_limits_file'])
            identity['actual_action_bounds']=result.get('actual_action_bounds')
            result['frozen_interface_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            (output/'interface.json').write_text(json.dumps(identity,indent=2,allow_nan=False))
            runtime_identity=runtime_identity_payload(config,result,identity)
            (output/'runtime_identity.json').write_text(json.dumps(runtime_identity,indent=2,allow_nan=False))
            result['runtime_identity_path']=str(output/'runtime_identity.json')
            result['runtime_identity_sha256']=digest(output/'runtime_identity.json')
            result['video_encoding']=runtime_identity['video_encoding']
            captures=output/'policy_observations';captures.mkdir()
            png=output/'head_rgb'
            if config.get('save_full_rgb'):png.mkdir()
            video=imageio.get_writer(str(output/'head_50fps.mp4'),fps=50,codec='libx264',macro_block_size=None,output_params=list(VIDEO_OUTPUT_PARAMS))
            observer_video=imageio.get_writer(str(output/'observer_50fps.mp4'),fps=50,codec='libx264',macro_block_size=None,output_params=list(VIDEO_OUTPUT_PARAMS))
            if config['mode']=='policy':model=PolicyProcess(config['policy_command'],output/'policy.stderr.log',config.get('policy_timeout_seconds',180))
            reset_counter=int(raw._sim_step_counter);chunk=[];chunk_at=0;last32=last_raw=last50=None
            initial_hold=np.r_[measured(robot.data.joint_pos)[upper_ids],.75,0.,0.,0.].astype(np.float32)
            replay=None
            if config['mode']=='replay':
                replay_path=Path(config['replay_actions_npy'])
                expected=config.get('replay_actions_sha256',config.get('replay_actions_npy_sha256'))
                if not expected or digest(replay_path)!=expected:raise ValueError('Replay input must match explicit SHA')
                result['replay_actions_sha256']=expected
                replay=np.asarray(np.load(replay_path,allow_pickle=False),dtype=np.float32)
                if replay.ndim!=2 or replay.shape[1]!=32 or len(replay)<config['ticks'] or not np.isfinite(replay).all():raise ValueError('Replay must supply every finite32 action; no implicit tail hold')
            with (output/'trajectory.jsonl').open('x') as trace:
                for tick in range(config['ticks']+1):
                    check_wall_deadline(config,result)
                    counters=(int(raw._sim_step_counter),int(raw.sim._physics_step_count))
                    q=measured(robot.data.joint_pos).copy();rgb=measured(camera.data.output['rgb'])[:,:,:3].copy()
                    if rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8:raise ValueError('Original640x480RGB changed')
                    observer_rgb=measured(observer.data.output['rgb'])[:,:,:3].copy()
                    target=term.processed_actions[0].detach().cpu().numpy()
                    row={'control_tick':tick,'physics_steps_since_reset':counters[0]-reset_counter,
                        'monotonic_ns':time.monotonic_ns(),'wall_time_ns':time.time_ns(),'sim_time_seconds':tick/50.,
                        'source_joint_names':names,'joint_positions':q.tolist(),'joint_velocities':measured(robot.data.joint_vel).tolist(),
                        'root_link_pose_w':measured(robot.data.root_link_pose_w).tolist(),
                        'root_link_velocity_w':measured(robot.data.root_link_vel_w).tolist(),'root_com_velocity_w':measured(robot.data.root_com_vel_w).tolist(),
                        'upright':float(-measured(robot.data.projected_gravity_b)[2]),
                        'acceptance_truth_only':{n:{'pose':measured(raw.scene[n].data.root_pose_w).tolist(),'velocity':measured(raw.scene[n].data.root_vel_w).tolist()} for n in OBJECTS},
                        'acceptance_contacts_only':sample_contacts(contacts,tensor,contact_meta),
                        'acceptance_robot_body_link_poses_w':measured(robot.data.body_link_pose_w).tolist(),
                        'processed_joint_targets':target.tolist(),'processed_upper28_targets':target[upper_ids].tolist(),
                        'processed_rest15_targets':target[rest_ids].tolist(),'rest15_names':[names[i] for i in rest_ids],
                        'issued_action32_raw':last_raw,'applied_action32':last32,'applied_source_action50':last50,
                        'original_terms':copy.deepcopy(events),'head_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                        'observer_rgb_sha256':hashlib.sha256(observer_rgb.tobytes()).hexdigest()}
                    if preflight_telemetry is not None:
                        row['acceptance_preflight_only']=preflight_telemetry.sample(measured,observer)
                    if counters!=(int(raw._sim_step_counter),int(raw.sim._physics_step_count)) or row['physics_steps_since_reset']!=tick*4:
                        raise ValueError('Telemetry advanced physics or loop reset')
                    if tick==0:
                        initial={k:row[k] for k in ('source_joint_names','joint_positions','joint_velocities','root_link_pose_w','root_link_velocity_w','root_com_velocity_w','acceptance_truth_only')}
                        result['initial_physical_state_schema']='pi05_g1_mobile_initial_kinematic_v1'
                        result['initial_physical_state_sha256']=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest()
                    video.append_data(rgb);observer_video.append_data(observer_rgb)
                    if config.get('save_full_rgb'):Image.fromarray(rgb).save(png/f'{tick:05}.png')
                    rows.append(row);trace.write(json.dumps(row,allow_nan=False)+'\n');trace.flush()
                    result['completed_control_ticks']=tick
                    if tick%50==0 or tick==config['ticks']:
                        print(json.dumps({'event':'mobile_progress','case_id':case['case_id'],
                            'control_tick':tick,'sim_seconds':tick/50.,'policy_calls':len(result['inference_calls']),
                            'upright':row['upright']},separators=(',',':')),flush=True)
                    # Preserve original source fall guard relative to its floor.
                    # The original drop predicate is only a termination reason;
                    # root's independent acceptance determines final ground-drop.
                    floor_z=float(config.get('scene',{}).get('floor_z_m',-.795))
                    if row['root_link_pose_w'][2]-floor_z<.35 or row['upright']<.5:
                        result.update(physical_failure=True,termination_reason='source_fall_guard')
                        break
                    if events.get('object_dropped',{}).get('ever'):
                        result.update(physical_failure=True,termination_reason='source_object_dropped_predicate')
                        break
                    if config.get('stop_on_verified_success') is True and config['mode']=='policy' and config.get('evaluate_acceptance') is True:
                        from experiments.pi05_g1_mobile.acceptance import evaluate_episode
                        check=verified_success_check(rows,audit_context,evaluate_episode)
                        if check is not None:
                            result['external_success_audits'].append(check)
                            if check['success']:
                                result.update(termination_reason='verified_task_success',external_audit_terminated_episode=True)
                                break
                    if tick==config['ticks']:break
                    if config['mode']=='diagnostic':action32=initial_hold.copy()
                    elif config['mode']=='replay':action32=replay[tick].copy()
                    else:
                        if chunk_at==min(len(chunk),config['replan_interval_ticks']):
                            state=encode_state(q,names)
                            Image.fromarray(rgb).save(captures/f'head_{tick:05}.png')
                            (captures/f'observation_{tick:05}.json').write_text(json.dumps({'control_tick':tick,'state':state.tolist(),'measured_joint_positions43':q.tolist(),'measured_joint_names43':names,'prompt':config['prompt'],'rgb_sha256':row['head_rgb_sha256']},allow_nan=False))
                            chunk,timing=model.infer(rgb,state,case,tick,config['prompt'],q,names)
                            if chunk.shape!=(50,32):raise ValueError('Native50x32 reply required')
                            if counters!=(int(raw._sim_step_counter),int(raw.sim._physics_step_count)):raise ValueError('Policy RPC advanced physics')
                            np.save(captures/f'actions_{tick:05}.npy',chunk,allow_pickle=False)
                            result['inference_calls'].append(timing);chunk_at=0
                            check_wall_deadline(config,result)
                        action32=chunk[chunk_at].copy();chunk_at+=1
                    last_raw=action32.tolist()
                    if limits is not None:action32=np.clip(action32,*limits)
                    result['clipped_channels']=result.get('clipped_channels',0)+int(np.count_nonzero(action32!=np.asarray(last_raw,dtype=np.float32)))
                    result['commanded_channels']=result.get('commanded_channels',0)+32
                    action=np.asarray(decode_to_source(action32,q,names),dtype=np.float32)
                    if action.shape!=(50,) or not np.isfinite(action).all():raise ValueError('Decoder must return finite source50')
                    last32,last50=action32.tolist(),action.tolist()
                    check_wall_deadline(config,result)
                    started=time.monotonic_ns()
                    _,_,terminated,truncated,_=env.step(torch.as_tensor(action[None],device=raw.device))
                    result.setdefault('control_step_wall_seconds',[]).append((time.monotonic_ns()-started)/1e9)
                    if bool(terminated.any()) or bool(truncated.any()):raise ValueError('Unexpected source auto-reset')
            result.update(status='completed',actual_sdk_integrations=int(raw._sim_step_counter)-reset_counter)
            # Root owns task-specific acceptance. Diagnostic completion never claims task success.
            if config.get('evaluate_acceptance'):
                from experiments.pi05_g1_mobile.acceptance import evaluate_episode
                result['acceptance']=evaluate_episode(rows,audit_context)
                result['success']=bool(result['acceptance']['success'])
            finish()
    except BaseException as exc:
        finish(exc)
        raise
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();run(json.loads(args.config.read_text()))


if __name__=='__main__':main()
