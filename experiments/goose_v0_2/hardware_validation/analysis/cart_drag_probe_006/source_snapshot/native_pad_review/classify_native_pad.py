"""Read-only native TPU working-face classifier. No engine, steps, forces or state edits.

IMPORTANT: data.xpos/xmat must describe the SAME pose that generated contact.frame/
contact.pos/contact.dist and mj_contactForce. Never pass post-step poses to old contacts.
"""
import json,math
from pathlib import Path

DEFAULT_DOMAINS=Path(__file__).with_name('native_pad_domains.json')

def load_domains(path=DEFAULT_DOMAINS):
    return json.loads(Path(path).read_text())

def _R(R):
    if hasattr(R,'tolist'):R=R.tolist()
    return [R[i:i+3] for i in (0,3,6)] if len(R)==9 else R

def _dot(a,b):return sum(x*y for x,y in zip(a,b))
def _mv(R,v):return [_dot(r,v) for r in R]
def _tv(R,v):return [_dot([R[j][i] for j in range(3)],v) for i in range(3)]

def classify_point(side,point_world,body_pos_world,body_R_world_from_body,*,domains=None,
                   normal_from_pad_to_other_world=None,plane_tol_m=.0005,edge_band_m=.0005,
                   min_normal_cos=math.cos(math.radians(30))):
    """side='upper'/'lower'; positions SI, R maps owner body -> world.

    Geometry only. Thickness/penetration, friction, actual force and timestamp validity
    remain independent gates. Normal must point from pad toward the other object.
    """
    domains=load_domains() if domains is None else domains
    pad=next(x for x in domains['pads'] if x['side']==side)
    R=_R(body_R_world_from_body);q=_tv(R,[point_world[i]-body_pos_world[i] for i in range(3)])
    anchor=pad['plane_point_body_m'];normal=pad['outward_normal_body'];sd=_dot(normal,[q[i]-anchor[i] for i in range(3)])
    poly=pad['surface_polygon_uv_m'];margins=[]
    for a,b in zip(poly,poly[1:]+poly[:1]):
        e=[b[0]-a[0],b[1]-a[1]];margins.append((e[0]*(q[1]-a[1])-e[1]*(q[0]-a[0]))/math.hypot(*e))
    margin=min(margins);nc=None
    if normal_from_pad_to_other_world is not None:
        nn=list(normal_from_pad_to_other_world);ln=math.sqrt(_dot(nn,nn))
        nc=None if ln==0 else _dot(_mv(R,normal),nn)/ln
    if abs(sd)>plane_tol_m:label='OFF_PAD_WORKING_PLANE'
    elif margin < -edge_band_m:label='OUTSIDE_NATIVE_PAD_FOOTPRINT'
    elif margin < edge_band_m:label='PAD_EDGE_BAND_AMBIGUOUS'
    elif nc is None:label='PAD_FACE_POSITION_ONLY_NORMAL_UNCHECKED'
    elif nc < min_normal_cos:label='NON_FACE_CONTACT_NORMAL'
    else:label='NATIVE_PAD_FACE_INTERIOR'
    return {'side':side,'owner_body':pad['owner_body'],'classification':label,
        'geometry_verified_native_tpu_face':label=='NATIVE_PAD_FACE_INTERIOR',
        'point_body_m':q,'signed_distance_outward_plane_m':sd,'inward_edge_margin_m':margin,
        'normal_alignment_cos':nc,'plane_tolerance_m':plane_tol_m,'edge_band_m':edge_band_m,
        'scope':'working-face geometry only; not a friction/force/holding/numerical-validity pass'}

def classify_contact(model,data,contact,*,domains=None,plane_tol_m=.0005,edge_band_m=.0005):
    """Classify all upper/lower roles in one actual MuJoCo contact, without forward.

    Native header: contact.pos is midpoint, frame[0:3] points geom0 -> geom1.
    Reconstruct the pad proxy's own surface point, rather than testing a penetrating
    midpoint against a zero-thickness pad plane. Returns [] for other robot geoms.
    Caller must separately group by other_geom (e.g. cart_bite_bar versus yoke/shell)
    and preserve shell/edge/ambiguous force totals instead of silently dropping them.
    """
    domains=load_domains() if domains is None else domains
    gids=list(contact.geom) if hasattr(contact,'geom') else [int(contact.geom1),int(contact.geom2)]
    names=[model.geom(int(g)).name for g in gids];n=list(contact.frame[:3]);mid=list(contact.pos);dist=float(contact.dist);out=[]
    for k,name in enumerate(names):
        matches=[p for p in domains['pads'] if p['runtime_geom']==name]
        if not matches:continue
        pad=matches[0];bid=model.body(pad['owner_body']).id
        # p0=mid-dist*n/2, p1=mid+dist*n/2; dist may be negative.
        p=[mid[j]+(-.5 if k==0 else .5)*dist*n[j] for j in range(3)]
        nw=n if k==0 else [-x for x in n]
        row=classify_point(pad['side'],p,data.xpos[bid],data.xmat[bid],domains=domains,
            normal_from_pad_to_other_world=nw,plane_tol_m=plane_tol_m,edge_band_m=edge_band_m)
        row.update(pad_geom=name,other_geom=names[1-k],contact_midpoint_world_m=mid,
            reconstructed_pad_surface_point_world_m=p,contact_signed_distance_m=dist,
            penetration_exceeds0p5mm=dist<-.0005,
            expected_handle_pair=names[1-k]=='cart_bite_bar',
            requires_force_timestamp_matching=True)
        out.append(row)
    return out
