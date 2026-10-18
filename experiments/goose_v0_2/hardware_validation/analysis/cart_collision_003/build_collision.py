"""Cart collision first candidate: CPU compile/forward/render only; no integrations.

Run with the recorded MuJoCo 3.10 environment. All geometry is cart-frame SI.
CAD part masses, COMs and complete inertias remain authoritative.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MUJOCO_GL']='osmesa'
os.environ['GALLIUM_DRIVER']='softpipe'
os.environ['LIBGL_ALWAYS_SOFTWARE']='1'
import copy, hashlib, json, math, resource, time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull

START=time.time()
resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
try: os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
except AttributeError: pass
HERE=Path(__file__).resolve().parent
L=HERE.parent.parent
A=L/'assets/low_profile_stock_cart_02'
W=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots')
C=W/'robots/goose_v0_2/evidence/low_profile_stock_cart_02'
R=L/'inputs/physics_v0_2_20261018_04'
P=W/'robots/goose_v0_2/cad/source/low_profile_stock_cart_02/parameters.json'
EXPORT=W/'robots/goose_v0_2/cad/exports/low_profile_stock_cart_02/goose_low_profile_stock_cart_02'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def vec(v): return ' '.join(format(float(x),'.17g') for x in v)
def elm(p,t,**kw): return ET.SubElement(p,t,{k:str(v) for k,v in kw.items()})

def aggregate(parts, origin, reserve=False):
    rows=[]
    for q in parts:
        rows.append((q['mass_kg'],np.array(q['COM_mm'])*.001,np.array(q['full_inertia_about_COM_kg_m2'])))
    if reserve:
        # Explicit unmodelled fasteners/locks reserve, uniformly distributed over frame.
        m=.35; dims=np.array([.65,.45,.03]); c=np.array([.260,0,.180])
        I=m/12*np.diag([dims[1]**2+dims[2]**2,dims[0]**2+dims[2]**2,dims[0]**2+dims[1]**2])
        rows.append((m,c,I))
    mass=sum(x[0] for x in rows)
    com=sum(m*c for m,c,_ in rows)/mass
    inertia=sum(I+m*(np.dot(c-com,c-com)*np.eye(3)-np.outer(c-com,c-com)) for m,c,I in rows)
    return {'mass_kg':mass,'COM_cart_m':com.tolist(),'COM_body_m':(com-origin).tolist(),
            'full_inertia_about_COM_kg_m2':inertia.tolist(),'source_parts':[p['name'] for p in parts],
            'reserve_kg':.35 if reserve else 0,'minimum_principal_inertia':float(np.linalg.eigvalsh(inertia).min())}

def inertial(body, rec):
    I=np.array(rec['full_inertia_about_COM_kg_m2'])
    elm(body,'inertial',mass=rec['mass_kg'],pos=vec(rec['COM_body_m']),
        fullinertia=vec([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))

def rings_x(spec, n=16):
    return np.array([[x,r*math.cos(a),.480+r*math.sin(a)] for x,r in spec for a in np.arange(n)*2*math.pi/n])

def circle_on_axis(center,axis,r,n):
    axis=np.array(axis,dtype=float);axis/=np.linalg.norm(axis)
    u=np.cross(axis,[0,0,1] if abs(axis[2])<.9 else [0,1,0]);u/=np.linalg.norm(u);v=np.cross(axis,u)
    return np.array([center+r*(math.cos(a)*u+math.sin(a)*v) for a in np.arange(n)*2*math.pi/n])

def yoke_points(y):
    a=np.array([-.287,0,.480]);b=np.array([-.350,y,.480]);axis=b-a
    # 24 tube-ring + 32 support-ring points; no bridge between the two yokes.
    return np.vstack([circle_on_axis(a,axis,.007,12),circle_on_axis(b,axis,.007,12),
          circle_on_axis(b+[0,-.004,0],[0,1,0],.011,16),circle_on_axis(b+[0,.004,0],[0,1,0],.011,16)])

def main():
    params=json.loads(P.read_text()); screen=json.loads((C/'geometry_mass_screen.json').read_text())
    context=json.loads((C/'goose_cart_context.json').read_text())
    identity={'parameters':{'path':str(P),'sha256':sha(P)},'mass_screen':{'path':str(C/'geometry_mass_screen.json'),'sha256':sha(C/'geometry_mass_screen.json')},
       'context':{'path':str(C/'goose_cart_context.json'),'sha256':sha(C/'goose_cart_context.json')},
       'step':{'path':str(EXPORT.with_suffix('.step')),'sha256':sha(EXPORT.with_suffix('.step'))},
       'glb':{'path':str(EXPORT.with_suffix('.glb')),'sha256':sha(EXPORT.with_suffix('.glb'))}}
    baseline={str(p.relative_to(R)):sha(p) for p in sorted(R.rglob('*')) if p.is_file()}
    parts=copy.deepcopy(screen['parts']); original_parts=copy.deepcopy(parts); assert len(parts)==122
    theta=math.radians(params['drawhead_pitch_deg']);pivot=np.array(params['drawhead_pivot_center'])*.001;pz=float(pivot[2]);ct=math.cos(theta);st=math.sin(theta);Ry=np.array([[ct,0,st],[0,1,0],[-st,0,ct]])
    groups={'cart_base':[],'cart_yaw':[],'cart_pitch':[],'cart_roll':[]}
    origins={'cart_base':np.zeros(3),'cart_yaw':np.array([-.062,0,.480]),'cart_pitch':np.array([-.095,0,.480]),'cart_roll':np.array([-.224,0,.480])}
    for x in [80,570]:
        for y in [-190,190]:
            tag=f'{x}_{y}';groups['wheel_'+tag]=[];origins['wheel_'+tag]=np.array([x,y,62.5])*.001
            if x==80:groups['caster_'+tag]=[];origins['caster_'+tag]=np.array([55,y,135])*.001
    for p in parts:
        n=p['name']; g='cart_base'
        if n.startswith(('PU_tread_','nylon_spoked_hub_','wheel_6000_interface_')):
            for tag in ['80_-190','80_190','570_-190','570_190']:
                if tag in n:g='wheel_'+tag;break
        elif n.startswith(('fork_plate_80_','fork_bridge_80_','axle_d10_80_','caster_thrust_bearing_80_')) or n.startswith('caster_thrust_seat_80_') and n.endswith('129.5'):
            g='caster_80_-190' if '80_-190' in n else 'caster_80_190'
        elif n.startswith(('pitch_clevis_','yaw_rotor_to_pitch_','pitch_pin_','pitch_balance_spring_')):g='cart_yaw'
        elif n.startswith(('drawbar_main_','roll_bearing_housing','roll_bearing_608_')):g='cart_pitch'
        elif n.startswith(('rotating_forward_shaft_','shaft_axial_retaining_','bite_yoke_','bite_core_support_','bite_304_','replaceable_bite_')):g='cart_roll'
        if g in ['cart_pitch','cart_roll']:
            co=np.array(p['COM_mm'])*.001;I=np.array(p['full_inertia_about_COM_kg_m2']);p['COM_mm']=((pivot+Ry.T@(co-pivot))*1000).tolist();p['full_inertia_about_COM_kg_m2']=(Ry.T@I@Ry).tolist()
        groups[g].append(p)
    for g in ['cart_yaw','cart_pitch','cart_roll']:origins[g][2]=pz
    mci={g:aggregate(rows,origins[g],g=='cart_base') for g,rows in groups.items()}
    assert all(v['mass_kg']>0 and v['minimum_principal_inertia']>0 for v in mci.values())
    assert abs(sum(v['mass_kg'] for v in mci.values())-screen['estimated_empty_mass_kg'])<1e-10
    save(A/'source_identity.json',identity)
    save(A/'body_mass_map.json',{'status':'CAD_MCI_PLUS_DECLARED_RESERVE_NOT_MEASURED','body_map':mci,'frame_definition':'Body-zero axes are unpitched. Source02 initial downstream COM/inertia inverse-rotated by10deg about true pitch axis before aggregation. low_handle_420_pitch10 keyframe reconstructs actual CAD pose.',
       'source_initial_parts':original_parts,'inverse_CAD_initial_rotation_R':Ry.T.tolist(),'pitch_pivot_cart_m':pivot.tolist(),
       'reserve_assumption':{'mass_kg':.35,'COM_cart_m':[.260,0,.180],'inertia_model':'uniform 650x450x30mm distribution of unresolved fasteners and locks on frame'},
       'bearing_partition_assumption':'Entire wheel bearing envelopes assigned to wheel spin; actual inner/outer ring mass partition and caster bearing inertia require procurement readback.',
       'spring_partition_assumption':'Pitch balance spring envelope mass assigned to yaw fork. No stiffness/preload is invented.'})
    # 18 contact leaves. Boxes preserve the open cargo cavity, not decorative slots.
    specs=[]
    def geo(name,body,typ,pos,size,color,source,**kw):
        specs.append(dict(name=name,body=body,type=typ,pos=np.array(pos,float),size=size,color=color,source=source,**kw))
    blue=[.21,.56,.84,1];green=[.32,.70,.49,1];gray=[.48,.56,.61,1];orange=[.94,.50,.20,1];yellow=[.98,.74,.17,1]
    geo('cart_deck_frame','cart_base','box',[.325,0,.16575],[.325,.225,.01575],green,[p['name'] for p in parts[:7]])
    for y in [-.217,.217]:geo('cart_side_'+('left' if y>0 else 'right'),'cart_base','box',[.325,y,.23575],[.325,.008,.05425],blue,['basket side and posts'],approx='perforations closed; inner y=209mm retained')
    for x in [.008,.642]:geo('cart_end_'+('front' if x<.1 else 'rear'),'cart_base','box',[x,0,.23575],[.008,.209,.05425],blue,['basket end and posts'],approx='perforations closed; inner x=16..634mm retained')
    for x in [80,570]:
        for y in [-190,190]:
            tag=f'{x}_{y}';geo('cart_wheel_'+tag,'wheel_'+tag,'cylinder',origins['wheel_'+tag],[.0625,.0125],[.18,.23,.28,1],['PU_tread_'+tag,'nylon_spoked_hub_'+tag],axis=[0,1,0])
    geo('cart_mast','cart_base','box',[-.025,0,.350],[.016,.016,.200],gray,['height_adjust_mast_32x32x2'],approx='height pin holes closed; revised lower gussets not expanded into cargo cavity')
    geo('cart_yaw_housing','cart_base','cylinder',[-.062,0,.480],[.015,.022],gray,['yaw_housing'],axis=[0,0,1])
    for y in [-.017,.017]:geo('cart_pitch_clevis_'+('left' if y>0 else 'right'),'cart_yaw','box',[-.090,y,.480],[.021,.002,.0175],orange,['pitch_clevis_'+str(int(y*1000))],approx='filled pin hole only')
    geo('cart_drawbar','cart_pitch','mesh',[0,0,0],None,orange,['drawbar_main_22x2','roll_bearing_housing'],points=rings_x([(-.245,.017),(-.203,.017),(-.203,.011),(-.095,.011)]),approx='convex join across 6mm radius step; hollow tube interior filled')
    geo('cart_rotating_shaft','cart_roll','mesh',[0,0,0],None,orange,['rotating_forward_shaft_d8','bite_yoke_shaft_socket'],points=rings_x([(-.305,.004),(-.299,.007),(-.275,.007),(-.245,.004)]),approx='R4 exposed shaft ends and R7 local socket retained; one convex hull tapers across CAD steps; hollow socket and concealed shaft filled')
    for y in [-.085,.085]:geo('cart_bite_yoke_'+('left' if y>0 else 'right'),'cart_roll','mesh',[0,0,0],None,orange,['bite_yoke_tube_'+str(int(y*1000)),'bite_core_support_'+str(int(y*1000))],points=yoke_points(y),approx='each arm convexified separately including end support; hollow interiors filled')
    geo('cart_bite_bar','cart_roll','cylinder',[-.350,0,.480],[.006,.080],yellow,['replaceable_bite_sleeve_d12','bite_304_core_d8'],axis=[0,1,0])
    # Reference geoms are in unpitched body-zero frame, then initial keyframe applies actual10deg.
    for g in specs:
        if g['body'] in ['cart_yaw','cart_pitch','cart_roll'] or g['name']=='cart_yaw_housing':
            if g['type']=='mesh':g['points'][:,2]+=pz-.480
            else:g['pos'][2]+=pz-.480
        if g['name']=='cart_mast':g['pos'][2]=(params['mast_bottom_z']+params['mast_top_z'])*.0005;g['size'][2]=(params['mast_top_z']-params['mast_bottom_z'])*.0005
    assert len(specs)==18,len(specs)
    meshfiles={}
    for s in specs:
        if s['type']=='mesh':
            points=s.pop('points');h=ConvexHull(points);tm=trimesh.Trimesh(vertices=points,faces=h.simplices,process=True).convex_hull
            assert len(tm.vertices)<=64 and tm.is_convex and tm.is_watertight
            (A/'meshes').mkdir(exist_ok=True)
            f='meshes/'+s['name']+'.obj';(A/f).write_text(tm.export(file_type='obj'));meshfiles[s['name']]=f
    def scene(combined=False):
        if combined:
            root=ET.parse(R/'robot.xml').getroot()
            for e in root.findall('asset/mesh'):e.set('file','../../inputs/physics_v0_2_20261018_04/'+e.get('file'))
            root.find('worldbody/geom[@name="ground"]').set('conaffinity','6')
            world=root.find('worldbody');asset=root.find('asset')
        else:
            root=ET.Element('mujoco',model='low_profile_stock_cart_02_collision_first')
            elm(root,'compiler',angle='radian',inertiafromgeom='false',balanceinertia='false')
            elm(root,'option',timestep='.005',gravity='0 0 -9.81',integrator='implicitfast')
            asset=elm(root,'asset');world=elm(root,'worldbody')
            elm(world,'geom',name='ground',type='plane',size='3 3 .01',contype='1',conaffinity='4',rgba='.96 .97 .98 1')
        elm(asset,'texture',name='cart_white_sky',type='skybox',builtin='gradient',rgb1='1 1 1',rgb2='1 1 1',width='512',height='512')
        visual=elm(root,'visual');elm(visual,'global',offwidth='1600',offheight='1200');elm(visual,'quality',shadowsize='0')
        elm(visual,'headlight',ambient='.3 .3 .3',diffuse='.4 .4 .4',specular='.05 .05 .05')
        elm(world,'light',pos='0 -1 2',dir='0 0 -1',directional='true',castshadow='false',diffuse='.35 .35 .35',ambient='.1 .1 .1')
        elm(world,'light',pos='1 1 1.8',dir='0 0 -1',directional='true',castshadow='false',diffuse='.25 .25 .25',ambient='.05 .05 .05')
        bodies={}
        base=elm(world,'body',name='cart_base',pos=vec(np.array(params['cart_origin_robot_frame_mm'])*.001) if combined else '0 0 0');elm(base,'freejoint',name='cart_root');bodies['cart_base']=base
        for x in [80,570]:
            for y in [-190,190]:
                tag=f'{x}_{y}';parent='cart_base'
                if x==80:
                    name='caster_'+tag;origin=origins[name];b=elm(base,'body',name=name,pos=vec(origin));elm(b,'joint',name='cart_swivel_'+tag,type='hinge',axis='0 0 1',limited='false',damping='0',frictionloss='0');bodies[name]=b;parent=name
                name='wheel_'+tag;b=elm(bodies[parent],'body',name=name,pos=vec(origins[name]-origins[parent]));elm(b,'joint',name='cart_wheel_spin_'+tag,type='hinge',axis='0 1 0',limited='false',damping='0',frictionloss='0');bodies[name]=b
        parent='cart_base'
        for name,axis,limit in [('cart_yaw','0 0 1',[-math.pi/3,math.pi/3]),('cart_pitch','0 1 0',[-math.radians(25),math.radians(25)]),('cart_roll','1 0 0',None)]:
            b=elm(bodies[parent],'body',name=name,pos=vec(origins[name]-origins[parent]));kw=dict(name=name+'_hinge',type='hinge',axis=axis,damping='0',frictionloss='0',limited='true' if limit else 'false')
            if limit:kw['range']=vec(limit)
            elm(b,'joint',**kw);bodies[name]=b;parent=name
        for g,b in bodies.items():inertial(b,mci[g])
        for s in specs:
            kw=dict(name=s['name'],type=s['type'],rgba=vec(s['color']),density='0',contype='4',conaffinity='7',condim='3',friction='.65 .01 .002',solref='.005 1',margin='0')
            if s['type']=='mesh':
                elm(asset,'mesh',name=s['name'],file=meshfiles[s['name']],maxhullvert='64');kw['mesh']=s['name'];kw['pos']=vec(-origins[s['body']])
            else:
                kw['pos']=vec(s['pos']-origins[s['body']]);kw['size']=vec(s['size'])
                if s['type']=='cylinder':
                    quat=np.zeros(4);mj.mju_quatZ2Vec(quat,np.array(s.get('axis',[0,0,1]),float));kw['quat']=vec(quat)
            elm(bodies[s['body']],'geom',**kw)
        initial_cart=[*(np.array(params['cart_origin_robot_frame_mm'])*.001 if combined else [0,0,0]),1,0,0,0,0,0,0,0,0,0,0,theta,0]
        assert len(initial_cart)==16
        key=elm(root,'keyframe');elm(key,'key',name='low_handle_420_pitch10',qpos=vec((params['robot_full_qpos'] if combined else [])+initial_cart))
        f=A/('goose_cart_scene.xml' if combined else 'cart.xml');ET.indent(root);ET.ElementTree(root).write(f,encoding='utf-8',xml_declaration=True)
        return f
    standalone=scene();combined=scene(True)
    model=mj.MjModel.from_xml_path(str(standalone));data=mj.MjData(model);mj.mj_resetDataKeyframe(model,data,0);mj.mj_forward(model,data)
    full=mj.MjModel.from_xml_path(str(combined));fd=mj.MjData(full);mj.mj_resetDataKeyframe(full,fd,0);mj.mj_forward(full,fd)
    original=mj.MjModel.from_xml_path(str(R/'robot.xml'))
    cartids=[i for i in range(full.ngeom) if mj.mj_id2name(full,mj.mjtObj.mjOBJ_GEOM,i).startswith('cart_')]
    robotids=[i for i in range(full.ngeom) if i not in cartids and int(full.geom_type[i])!=int(mj.mjtGeom.mjGEOM_PLANE)]
    assert len(cartids)==18 and len(robotids)==14
    for attr in ['body_mass','body_ipos','body_inertia','body_iquat','body_pos','body_quat']:
        assert np.array_equal(getattr(full,attr)[:original.nbody],getattr(original,attr)),attr
    for attr in ['jnt_axis','jnt_pos','jnt_range','jnt_limited']:
        assert np.array_equal(getattr(full,attr)[:original.njnt],getattr(original,attr)),attr
    for attr in ['actuator_gear','actuator_ctrlrange','actuator_forcerange','eq_data','eq_type']:
        assert np.array_equal(getattr(full,attr),getattr(original,attr)),attr
    geom_rows=[];glbscene=trimesh.Scene()
    typecounts={}
    for i in cartids:
        name=mj.mj_id2name(full,mj.mjtObj.mjOBJ_GEOM,i);typ=int(full.geom_type[i]);kind={6:'box',5:'cylinder',7:'mesh'}[typ];typecounts[kind]=typecounts.get(kind,0)+1
        row={'name':name,'type':kind,'body':mj.mj_id2name(full,mj.mjtObj.mjOBJ_BODY,int(full.geom_bodyid[i])),'enabled_contype':int(full.geom_contype[i]),'conaffinity':int(full.geom_conaffinity[i])}
        if kind=='box':tm=trimesh.creation.box(extents=full.geom_size[i]*2)
        elif kind=='cylinder':tm=trimesh.creation.cylinder(radius=full.geom_size[i,0],height=2*full.geom_size[i,1],sections=48)
        else:
            mid=int(full.geom_dataid[i]);a=int(full.mesh_vertadr[mid]);n=int(full.mesh_vertnum[mid]);v=full.mesh_vert[a:a+n].copy();h=ConvexHull(v);tm=trimesh.Trimesh(vertices=v,faces=h.simplices,process=True).convex_hull
            row.update(loaded_mesh_vertices=n,convex_support_vertices=len(tm.vertices),convex_hull_triangles=len(tm.faces),convex=bool(tm.is_convex),watertight=bool(tm.is_watertight))
            assert len(tm.vertices)<=64
        T=np.eye(4);T[:3,:3]=fd.geom_xmat[i].reshape(3,3);T[:3,3]=fd.geom_xpos[i]-np.array(params['cart_origin_robot_frame_mm'])*.001;tm.apply_transform(T)
        tm.visual.vertex_colors=(full.geom_rgba[i]*255).astype(np.uint8);glbscene.add_geometry(tm,node_name=name,geom_name=name)
        row['display_triangles']=len(tm.faces);row['display_triangles_are_runtime_collision_faces']=kind=='mesh'
        row['cart_world_bounds_m']=tm.bounds.tolist();geom_rows.append(row)
    (A/'cart_collision.glb').write_bytes(glbscene.export(file_type='glb'))
    # Actual engine rendering of enabled proxies: no artist/CAD substitution.
    def render(m,d,name,lookat,distance,azimuth):
        renderer=mj.Renderer(m,height=1050,width=1500);cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=lookat;cam.distance=distance;cam.azimuth=azimuth;cam.elevation=-22
        opt=mj.MjvOption();opt.flags[mj.mjtVisFlag.mjVIS_CONVEXHULL]=False
        renderer.update_scene(d,camera=cam,scene_option=opt);arr=renderer.render();renderer.close()
        image=Image.fromarray(arr);draw=ImageDraw.Draw(image)
        fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(fontpath,25)
        title='CART COLLISION 003: 18 + GOOSE 14 = 32' if name.startswith('goose') else 'CART COLLISION 003 | 18 contact leaves'
        draw.rectangle((0,0,1500,98),fill='white');draw.text((30,15),title,fill='#20313d',font=font)
        draw.text((30,51),'Compiled MuJoCo 3.10 proxies | CPU | 0 physics steps | Not drag-qualified',fill='#425963',font=font)
        image.save(HERE/name)
    render(model,data,'cart_collision.png',[.16,0,.26],1.65,145)
    render(full,fd,'goose_cart_collision.png',[.53,0,.30],2.40,145)
    # Static functional checks. Interior points are well clear of thin basket walls.
    def occupied(point):
        point=np.array(point); hits=[]
        for s in specs:
            if s['type']=='box':inside=bool(np.all(np.abs(point-s['pos'])<=np.array(s['size'])+1e-10))
            elif s['type']=='cylinder':
                q=point-s['pos'];ax=np.array(s.get('axis',[0,0,1]));along=np.dot(q,ax);rad=np.linalg.norm(q-along*ax);inside=abs(along)<=s['size'][1]+1e-10 and rad<=s['size'][0]+1e-10
            else:
                tm=trimesh.load(A/meshfiles[s['name']],force='mesh');h=ConvexHull(tm.vertices);inside=bool(np.all(h.equations[:,:3]@point+h.equations[:,3]<=1e-9))
            if inside:hits.append(s['name'])
        return hits
    cavity_points=[[x,y,z] for x in [.021,.325,.629] for y in [-.201,0,.201] for z in [.185,.235,.289]]
    cavity=[{'p_m':p,'hits':occupied(p)} for p in cavity_points];assert all(not x['hits'] for x in cavity)
    approach_points=[[x,y,z] for x in [-.38,-.35,-.32] for y in [-.04,0,.04] for z in [pz-.015,pz+.015]]
    approach=[{'p_m':p,'hits':occupied(p)} for p in approach_points];assert all(not x['hits'] for x in approach)
    wheel_checks=[]
    for x in [80,570]:
        for y in [-190,190]:wheel_checks.append({'wheel':f'{x}_{y}','radius_m':.0625,'width_m':.025,'neutral_lowest_z_m':0,'front_trail_m':.025 if x==80 else 0,'CAD_radius_error_m':0,'CAD_width_error_m':0})
    omitted=[p for p in parts if p['name'].startswith(('fork_plate_','fork_bridge_','parking_pedal_','parking_friction_pad_','caster_','load_spreader_','mount_plate_','axle_d10_'))]
    omissions=[{'name':p['name'],'bounds_cart_mm':p['bounds_mm'],'mass_retained_kg':p['mass_kg']} for p in omitted]
    stats={'status':'COLLISION_003_CANDIDATE_COMPILE_AND_FORWARD_ONLY','collision_candidate_id':'low_profile_stock_cart_02_collision_003','engine_version':mj.__version__,'cart_contact_leaves':len(cartids),'cart_types':typecounts,
      'robot_contact_leaves':len(robotids),'combined_robot_and_cart':len(cartids)+len(robotids),'ground_planes':1,'cargo_objects':0,'cargo_target_kg':20,
      'cart_bodies':10,'cart_passive_hinges':9,'cart_root_free_dofs':6,'cart_mass_kg':sum(v['mass_kg'] for v in mci.values()),
      'mesh_support_vertices_total':sum(r.get('convex_support_vertices',0) for r in geom_rows),'mesh_hull_triangles_total':sum(r.get('convex_hull_triangles',0) for r in geom_rows),
      'max_mesh_support_vertices':max(r.get('convex_support_vertices',0) for r in geom_rows),'analytical_shapes_have_no_runtime_triangle_count':True,'actual_initial_keyframe':'low_handle_420_pitch10',
      'geom_rows':geom_rows,'baseline_53_files_unchanged':baseline=={str(p.relative_to(R)):sha(p) for p in sorted(R.rglob('*')) if p.is_file()},
      'robot_compiled_MCI_axes_drive_equalities_unchanged':True,'physics_integrations':0,'training_updates':0,'GPU':0,
      'qualification':'Only actual compile, one neutral forward and specified robot FK. No stepping, rolling resistance, grip, backward control or 20kg capacity qualification.'}
    geom_bar=full.geom('cart_bite_bar').id
    target=np.array([.300,0,.420]);assert np.linalg.norm(fd.geom_xpos[geom_bar]-target)<1e-9
    clear={}
    for name in ['upper_bill_envelope','lower_bill_envelope']:
        for other in ['cart_bite_bar','cart_rotating_shaft','cart_bite_yoke_left','cart_bite_yoke_right']:
            ft=np.zeros(6);dd=mj.mj_geomDistance(full,fd,full.geom(name).id,full.geom(other).id,1,ft);clear[name+'__'+other]={'distance_m':float(dd),'points_world_m':ft.reshape(2,3).tolist()}
    cartbody=[i for i in range(full.nbody) if full.body_mass[i]>0 and i>=original.nbody]
    totalmass=float(full.body_mass[cartbody].sum());com=sum(full.body_mass[i]*fd.xipos[i] for i in cartbody)/totalmass
    II=np.zeros((3,3))
    for i in cartbody:
        Q=fd.ximat[i].reshape(3,3);v=fd.xipos[i]-com;II+=Q@np.diag(full.body_inertia[i])@Q.T+full.body_mass[i]*(np.dot(v,v)*np.eye(3)-np.outer(v,v))
    save(A/'initial_state.json',{'keyframe':'low_handle_420_pitch10','combined_qpos':fd.qpos.tolist(),'standalone_qpos':data.qpos.tolist(),'robot_full_qpos':fd.qpos[:29].tolist(),'actual_bar_world_m':fd.geom_xpos[geom_bar].tolist(),'actual_bar_cart_m':(fd.geom_xpos[geom_bar]-np.array(params['cart_origin_robot_frame_mm'])*.001).tolist(),'cart_origin_world_m':(np.array(params['cart_origin_robot_frame_mm'])*.001).tolist(),'cart_origin_units':'m','pitch_initial_rad':theta,'pitch_pivot_cart_m':pivot.tolist(),'yaw_pivot_cart_m':[-.062,0,pz],'contact_distances':clear,'cart_loaded_mass_kg':totalmass,'cart_loaded_COM_world_m':com.tolist(),'cart_loaded_full_inertia_about_COM_world_kg_m2':II.tolist(),'physics_integrations':0})
    assert abs(totalmass-screen['estimated_empty_mass_kg'])<1e-10
    save(HERE/'loaded_collision_stats.json',stats)
    save(HERE/'functional_checks.json',{'cargo_cavity_probe_count':len(cavity),'cargo_cavity':cavity,'mouth_approach_probe_count':len(approach),'mouth_approach':approach,'wheel_dimensions':wheel_checks,
       'deviation_scope':'Finite geometric probes, not complete Hausdorff bound or swept contact validation. Perforations/drain slots are intentionally closed; no internal cavity fill.',
       'frame_deck_overfill':'650x450x31.5mm slab preserves deck plane181.5mm and underframe low150mm; fills spaces between frame rails. Not qualified for underframe obstacles.',
       'side_overfill':'16mm side boxes at y±217 close decorative openings and expand sheet material to rim thickness; minimum inner cargo box is618x418mm, leaving600x400mm tote room.',
       'fork_contact_omissions':omissions,'fork_domain_limit':'Smooth floor only. Forks/plates/brakes have full CAD MCI but no individual contact leaves. No narrow obstacle, robot-foot near-fork, curb, threshold or all-heading collision qualification.',
       'fork_outer_side_vs_wheel_tread_max_extra_mm':27.5,'note_fork_extra':'Wheel outer |y|202.5; top spreaders/mount |y|230 =>27.5mm extra; axle reaches216 =>13.5mm. Overall omission list is source truth.',
       'unfrozen_dynamics':['actual wheel rolling and swivel resistance','bearing mass partition','tow joint friction and pitch balance spring stiffness/preload','parking brake released state material clearance','measured contact materials/softness','loaded payload geometry/MCI','free robot equilibrium/grip/stepping controller']})
    save(A/'collision_source_map.json',{'candidate':'low_profile_stock_cart_02_collision_003','geometry_source':identity,'cart_geoms':[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in s.items()} for s in specs],
      'contact_rules':'Cart contype4 conaffinity7, original robot contype2/conaffinity3 unchanged; cart receives robot/ground/cart. Ground conaffinity expanded2to6 only in derived scene. MuJoCo same-body and adjacent-body filters apply; no global cart self-collision disable or robot-cart binding.',
      'joint_assumptions':'Axes/limits from CAD expert, no motors; bearing drag/spring zero placeholders explicitly unqualified. Passive hinges are free for subsequent study, never locked for display.',
      'cargo':'No20kg cargo is instantiated in collision count or render; target is not a successful load test.'})
    save(HERE/'source_input_immutability.json',{'robot_files':baseline,'robot_all_unchanged':stats['baseline_53_files_unchanged'],'cart_source':identity,'cart_source_unchanged':all(sha(v['path'])==v['sha256'] for v in identity.values())})
    save(HERE/'resource_receipt.json',{'wall_seconds':time.time()-START,'cpu_affinity':sorted(os.sched_getaffinity(0)),'thread_limit_env':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},'address_space_limit_GiB':8,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'physics_steps':0,'forward_calls':2,'GPU':0,'training_updates':0,'process_exits_after_return':True})
    save(HERE/'artifact_manifest.json',{'files':[{'path':str(p.relative_to(L)),'bytes':p.stat().st_size,'sha256':sha(p)} for folder in [A,HERE] for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json']})
    print(json.dumps({k:stats[k] for k in ['cart_contact_leaves','cart_types','robot_contact_leaves','combined_robot_and_cart','cart_mass_kg','mesh_support_vertices_total','mesh_hull_triangles_total','max_mesh_support_vertices','physics_integrations']},indent=2))

if __name__=='__main__':main()
