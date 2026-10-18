"""Independent mobile physical task audit. Its truth is never a policy input.

Distance uses measured box motion after a two-hand contact/lift proxy. Supported
release uses original bin-floor collision geometry, fresh PhysX force/points and
conservative separation from every exported robot collider. A proximity event
alone cannot pass. These observations do not establish force closure or hardware.
"""
from __future__ import annotations
import hashlib,json,math,sys
from functools import lru_cache
from pathlib import Path
import numpy as np
SCHEMA='pi05_g1_mobile_acceptance_v1'


def _array(x,shape,name):
    x=np.asarray(x,dtype=np.float64)
    if x.shape!=shape or not np.isfinite(x).all():raise ValueError('Missing/nonfinite '+name)
    return x


def rotation_xyzw(pose):
    x,y,z,w=_array(pose,(7,),'actor pose')[3:]
    if abs(x*x+y*y+z*z+w*w-1)>1e-3:raise ValueError('Actor quaternion is not normalized')
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def _hull(points):
    pts=sorted(set(map(tuple,np.asarray(points)[:,:2])))
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower=[];upper=[]
    for p in pts:
        while len(lower)>1 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper)>1 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    out=np.asarray(lower[:-1]+upper[:-1],dtype=float)
    if len(out)<3:raise ValueError('Degenerate bin floor')
    return out


@lru_cache(maxsize=4)
def geometry(definition_path,expected_sha,robot_path,robot_sha,harness):
    def read(path,sha):
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Frozen acceptance geometry changed')
        return json.loads(raw)
    definition=read(definition_path,expected_sha);robot=read(robot_path,robot_sha)
    objects={item['kind']:item for item in definition['objects']}
    box=np.concatenate([part['points'] for part in objects['t2_box']['convex_parts']])
    floor=np.asarray(objects['t2_bin']['convex_parts'][2]['points'])
    polygon=_hull(floor);edge=np.roll(polygon,-1,axis=0)-polygon
    inward=np.column_stack((-edge[:,1],edge[:,0]));inward/=np.linalg.norm(inward,axis=1)[:,None]
    sys.path.insert(0,harness)
    from unitree_g1_t1_source_placement import collision_bounds,aabb_gap
    return box,floor,polygon,inward,robot,collision_bounds,aabb_gap


def stable_pose_window(rows,position_m=.005,angle_degrees=5.):
    """All pairs of box AND bin actor poses; avoids cumulative slow drift."""
    for actor in ('brown_box','blue_sorting_bin'):
        poses=np.asarray([r['acceptance_truth_only'][actor]['pose'] for r in rows],float)
        d=poses[:,None,:3]-poses[None,:,:3]
        if np.linalg.norm(d,axis=-1).max()>position_m:return False
        q=poses[:,3:];dots=np.clip(np.abs(q@q.T),0,1)
        if 2*np.arccos(dots).max()>math.radians(angle_degrees):return False
    return True


def evaluate_episode(rows,config):
    if not rows:raise ValueError('Cannot audit empty physical trace')
    box,floor,polygon,inward,robot,collision_bounds,aabb_gap=geometry(
        config['geometry_definition'],config['geometry_definition_sha256'],
        config['robot_geometry'],config['robot_geometry_sha256'],config['source_harness_dir'])
    rule=config['acceptance'];meta=config['audit_contact_metadata'];body_names=config['audit_body_names']
    if meta['schema_writes']!=0 or len(meta['robot_filter_indices'])!=53 or len(meta['hand_filter_indices'])!=16:
        raise ValueError('Foreign/incomplete contact identities')
    ids={b['name']:body_names.index(b['name']) for b in robot['bodies']}
    hz=50;required=round(float(rule['stable_seconds'])*hz)+1
    first=rows[0]['acceptance_truth_only'];initial_z=float(first['brown_box']['pose'][2])
    initial_position=np.asarray(first['brown_box']['pose'][:2]);target_position=np.asarray(first['blue_sorting_bin']['pose'][:2])
    task_distance=float(np.linalg.norm(target_position-initial_position))
    first_grasp=None;carry_origin=None;last_carry_tick=None;last_carried=None;carry_path=0.;max_carried=0.;carry_qualified_tick=None
    contact_gaps=0;largest_gap=0;gap=0;fell=False;dropped=False;suffix=[];max_stable=0.;qualified=None;samples=[];carry_yaws=[]
    for expected,row in enumerate(rows):
        tick=row['control_tick']
        if tick!=expected or row['physics_steps_since_reset']!=tick*4:raise ValueError('Trace reset/gap/extra physics')
        truth=row['acceptance_truth_only'];bp=_array(truth['brown_box']['pose'],(7,),'box pose');tp=_array(truth['blue_sorting_bin']['pose'],(7,),'bin pose')
        velocity=_array(truth['brown_box']['velocity'],(6,),'box velocity')
        rp=_array(row['root_link_pose_w'],(7,),'robot root');upright=float(row['upright'])
        c=row['acceptance_contacts_only'];hand=_array(c['box_hand_force_matrix_world_n'],(16,3),'hand contacts');rf=_array(c['box_robot_force_matrix_world_n'],(53,3),'robot contacts');bf=_array(c['box_bin_force_world_n'],(3,),'bin contacts')
        matrix=np.asarray(c['box_filter_force_matrix_world_n'],float)
        total_filters=len(meta['bin_filter_indices'])+len(meta['robot_filter_indices'])
        paths=meta['filter_paths']
        if len(paths)==1 and isinstance(paths[0],list):paths=paths[0]
        if len(paths)!=total_filters or matrix.shape!=(1,total_filters,3) or not np.isfinite(matrix).all():
            raise ValueError('Raw contact coverage mismatch')
        reconstructed=np.zeros_like(matrix[0]);robot_point=False;bin_point=False
        for p in c['points']:
            index=p['filter_index'];normal=_array(p['normal_world'],(3,),'contact normal');force=float(p['normal_force_n'])
            if type(index) is not int or not 0<=index<len(reconstructed) or not math.isfinite(force):raise ValueError('Malformed contact point')
            reconstructed[index]+=normal*force
            if index in meta['robot_filter_indices'] and (p['separation_m']<=0 or abs(force)>0):robot_point=True
            if index in meta['bin_filter_indices'] and abs(force)>0:bin_point=True
        if np.max(np.abs(reconstructed-matrix[0]))>1e-4:raise ValueError('Signed contact point/matrix disagreement')
        if not np.array_equal(rf,matrix[0,meta['robot_filter_indices']]) or not np.array_equal(hand,matrix[0,meta['hand_filter_indices']]):raise ValueError('Derived contact order differs')
        if np.max(np.abs(bf-matrix[0,meta['bin_filter_indices']].sum(0)))>1e-5:raise ValueError('Derived bin force differs')
        world=box@rotation_xyzw(bp).T+bp[:3];local=(world-tp[:3])@rotation_xyzw(tp)
        margin=float((local[:,:2]@inward.T-np.sum(polygon*inward,axis=1)).min())
        inside=margin>=0 and local[:,2].min()>=floor[:,2].min()-.002
        on_floor=local[:,2].min()<=floor[:,2].max()+.02
        contacts=int(np.count_nonzero(np.linalg.norm(hand,axis=1)>1e-6))
        left_contact=any(np.linalg.norm(rf[i])>1e-6 for i,name in enumerate(meta['robot_body_names']) if name.startswith('left_hand_'))
        right_contact=any(np.linalg.norm(rf[i])>1e-6 for i,name in enumerate(meta['robot_body_names']) if name.startswith('right_hand_'))
        held=left_contact and right_contact and bp[2]-initial_z>.03
        if held:
            if first_grasp is None:first_grasp=tick;carry_origin=bp[:2].copy()
            gap=0;max_carried=max(max_carried,float(np.linalg.norm(bp[:2]-carry_origin)))
            if last_carried is not None and tick-last_carry_tick<=round(.25*hz):carry_path+=float(np.linalg.norm(bp[:2]-last_carried))
            last_carried=bp[:2].copy();last_carry_tick=tick
            rot=rotation_xyzw(rp);carry_yaws.append(float(math.atan2(rot[1,0],rot[0,0])))
            span=float(np.ptp(np.unwrap(carry_yaws))) if carry_yaws else 0.
            if (carry_qualified_tick is None and task_distance>=rule['minimum_source_target_distance_m']
                    and max_carried>=rule['minimum_carried_displacement_m']
                    and span>=math.radians(rule['minimum_carry_turn_degrees'])):
                carry_qualified_tick=tick
        elif first_grasp is not None and not inside:
            if gap==0:contact_gaps+=1
            gap+=1;largest_gap=max(largest_gap,gap)
        root_height=float(rp[2])-float(config.get('source_ground_z',-.795))
        fell|=root_height<.35 or upright<.5
        ground_drop=first_grasp is not None and float(world[:,2].min())<=float(config.get('source_ground_z',-.795))+.03 and not inside
        dropped|=bool(ground_drop)
        standing=upright>.95 and root_height>=.35
        speed=float(np.linalg.norm(velocity[:3]));angular=float(np.linalg.norm(velocity[3:]));support=bool(bf[2]>0 and bin_point)
        separated=False;min_gap=None
        candidate=bool(inside and on_floor and support and not robot_point and not np.any(rf) and speed<.02 and angular<.1 and standing and not fell and not dropped)
        if candidate:
            poses=_array(row['acceptance_robot_body_link_poses_w'],(53,7),'measured body poses')
            bb=(world.min(0),world.max(0));gaps=[aabb_gap(bb,collision_bounds(collider,poses[ids[collider['body']]])) for collider in robot['collisions']]
            min_gap=float(min(gaps));separated=min_gap>0
        ready=candidate and separated
        suffix=(suffix+[row])[-required:] if ready else []
        if suffix:
            seconds=(len(suffix)-1)/hz
            max_stable=max(max_stable,seconds)
            if (len(suffix)==required and carry_qualified_tick is not None
                    and suffix[0]['control_tick']>=carry_qualified_tick
                    and stable_pose_window(suffix,rule['stable_position_m'],rule['stable_angle_degrees'])
                    and qualified is None):
                qualified={'start_tick':suffix[0]['control_tick'],'end_tick':tick,'seconds':float(rule['stable_seconds'])}
        samples.append({'tick':tick,'grasp_contact_lift_proxy':bool(held),'contacting_hand_bodies':contacts,'left_hand_contact':bool(left_contact),'right_hand_contact':bool(right_contact),
            'bin_floor_margin_m':margin,'box_inside_bin_prism':bool(inside),'bin_upward_force_n':float(bf[2]),
            'released_separation_proved':bool(separated),'minimum_robot_aabb_gap_m':min_gap,
            'box_speed_m_s':speed,'box_angular_speed_rad_s':angular,'standing':bool(standing),
            'fell':bool(fell),'ground_drop':bool(ground_drop),'placement_ready':bool(ready)})
    yaw_span=float(np.ptp(np.unwrap(carry_yaws))) if carry_yaws else 0.
    enough_distance=task_distance>=rule['minimum_source_target_distance_m'] and max_carried>=rule['minimum_carried_displacement_m']
    enough_turn=yaw_span>=math.radians(rule['minimum_carry_turn_degrees'])
    success=bool(first_grasp is not None and qualified is not None and enough_distance and enough_turn and not fell and not dropped)
    reasons=[]
    if first_grasp is None:reasons.append('no_verified_grasp')
    if not enough_distance:reasons.append('carry_distance_short')
    if not enough_turn:reasons.append('carry_turn_short')
    if qualified is None:reasons.append('no_released_stable_placement')
    if fell:reasons.append('fell')
    if dropped:reasons.append('ground_drop')
    end_tick=qualified['end_tick'] if success else rows[-1]['control_tick']
    return {'schema':SCHEMA,'success':success,'task_family':rule['task_family'],'grasped':first_grasp is not None,
        'first_grasp_control_tick':first_grasp,'carry_qualified_control_tick':carry_qualified_tick,'source_target_distance_m':task_distance,
        'max_carried_box_displacement_m':max_carried,'contact_qualified_carried_path_m':carry_path,
        'carry_turn_degrees':math.degrees(yaw_span),'hand_contact_gap_events_before_arrival':contact_gaps,
        'largest_hand_contact_gap_seconds':largest_gap/hz,'contact_gap_note':'Last200Hz solve sampled at50Hz; lack of sampled hand force alone is not a proven dropped object.',
        'stable_placement':qualified is not None,'qualified_stability_window':qualified,'max_ready_span_seconds':max_stable,
        'fell':fell,'dropped':dropped,'timeout':not success and not fell and not dropped,
        'termination_reasons':reasons,'task_sim_seconds':end_tick/hz,
        'task_wall_seconds':(rows[end_tick]['monotonic_ns']-rows[0]['monotonic_ns'])/1e9,
        'policy_external_interventions':0,'intervention_rule':'Policy-only runner admits no external navigation/actor-pose/control input; diagnostic/replay never count as autonomous trials.',
        'actual_control_ticks':rows[-1]['control_tick'],'physics_integrations':rows[-1]['physics_steps_since_reset'],
        'placement_rule':rule,'samples':samples}
