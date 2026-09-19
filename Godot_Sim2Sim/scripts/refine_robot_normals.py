"""Derive crease-preserving visual normals without changing source mesh geometry."""
import hashlib
import argparse
import json
import math
import re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PARTS={'right_shell','left_shell','top_head_shell','bottom_head_shell','upper_leg_left','leg','foot_left','foot_right','hip_l','neck','neck_pitch'}


def process(source,target,crease=42):
    lines=source.read_text().splitlines()
    vertex_lines=[line for line in lines if line.startswith('v ')]
    vertices=np.array([[float(x) for x in line.split()[1:4]] for line in vertex_lines])
    faces=np.array([[int(token.split('/')[0])-1 for token in line.split()[1:]] for line in lines if line.startswith('f ')],dtype=np.int32)
    assert faces.shape[1]==3
    points=vertices[faces]
    cross=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])
    length=np.linalg.norm(cross,axis=1)
    face_normals=cross/np.maximum(length[:,None],1e-20)
    angle=np.zeros((len(faces),3))
    for corner in range(3):
        a=points[:,(corner+1)%3]-points[:,corner]
        b=points[:,(corner+2)%3]-points[:,corner]
        angle[:,corner]=np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.einsum('ij,ij->i',a,b))
    # Join coincident CAD vertices for normal calculation only. Index/position data
    # written to the derived mesh remains identical to the immutable original.
    groups={}
    for face in range(len(faces)):
        for corner in range(3):
            key=tuple(np.round(points[face,corner],7))
            groups.setdefault(key,[]).append((face,corner))
    corner_normals=np.zeros((len(faces),3,3))
    threshold=math.cos(math.radians(crease))
    for entries in groups.values():
        ids=np.array([row[0] for row in entries]);corners=np.array([row[1] for row in entries])
        normals=face_normals[ids]
        compatible=normals@normals.T >= threshold
        weights=angle[ids,corners]
        blended=compatible@(normals*weights[:,None])
        blended/=np.maximum(np.linalg.norm(blended,axis=1)[:,None],1e-20)
        corner_normals[ids,corners]=blended
    normal_rows=[];normal_indices=np.empty((len(faces),3),dtype=np.int32);normal_map={}
    for face in range(len(faces)):
        for corner in range(3):
            normal=tuple(np.round(corner_normals[face,corner],8))
            if normal not in normal_map:
                normal_map[normal]=len(normal_rows)+1;normal_rows.append(normal)
            normal_indices[face,corner]=normal_map[normal]
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w') as stream:
        stream.write(f'# Visual derivative: original vertices and triangles, angle-weighted normals; crease {crease:g} degrees\n')
        stream.write('\n'.join(vertex_lines)+'\n')
        stream.writelines('vn '+' '.join(f'{v:.8f}' for v in row)+'\n' for row in normal_rows)
        stream.writelines('f '+' '.join(f'{v+1}//{n}' for v,n in zip(face,normals))+'\n' for face,normals in zip(faces,normal_indices))
    # Compare positions and triangle indices directly after serializing, including
    # winding. Normals are the only geometry attribute added by this operation.
    derived=target.read_text().splitlines()
    assert [line for line in derived if line.startswith('v ')]==vertex_lines
    derived_faces=np.array([[int(token.split('/')[0])-1 for token in line.split()[1:]] for line in derived if line.startswith('f ')])
    assert np.array_equal(derived_faces,faces)
    return {'source':str(source.relative_to(ROOT)),'target':str(target.relative_to(ROOT)),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'target_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'vertices':len(vertices),'triangles':len(faces),'normals':len(normal_rows),
            'positions_identical':True,'indices_and_winding_identical':True,'crease_degrees':crease}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--crease',type=float,default=42)
    parser.add_argument('--variant',default='')
    args=parser.parse_args()
    if not 1<=args.crease<=89:parser.error('Keep crease between 1 and 89 degrees.')
    if args.variant and not re.fullmatch('[a-z0-9_]+',args.variant):parser.error('Invalid variant name.')
    suffix='_'+args.variant if args.variant else ''
    output=ROOT/'godot/atelier'/('robot_normals'+suffix)
    roles=json.loads((ROOT/'godot/atelier/visual_mesh_roles.json').read_text())
    manifest=[]
    for robot,parts in roles.items():
        if robot not in {'microduck','microduck_roller'}:continue
        scene=(ROOT/'godot/generated'/robot/'robot.tscn').read_text()
        resources={match[2]:match[1] for match in re.findall(r'\[ext_resource type="([^"]+)" path="([^"]+)" id="([^"]+)"\]',scene)}
        mesh_by_geom={}
        for node in re.finditer(r'\[node name="vis_unnamed_(\d+)_\d+"[^\]]*\]\n(.*?)(?=\n\[|\Z)',scene,re.S):
            mesh=re.search(r'mesh = ExtResource\("([^"]+)"\)',node[2])
            if mesh:mesh_by_geom[node[1]]=resources[mesh[1]]
        seen={}
        for geom,part in parts.items():
            if part not in PARTS or geom not in mesh_by_geom:continue
            source=ROOT/'godot'/mesh_by_geom[geom].removeprefix('res://')
            target=output/robot/source.name
            if target in seen:
                seen[target]['geometry_ids'].append(geom)
                continue
            row=process(source,target,args.crease);row['part']=part;manifest.append(row)
            row['geometry_ids']=[geom];seen[target]=row
            print(robot,geom,part,row['triangles'],flush=True)
    (ROOT/'results/showcase').mkdir(parents=True,exist_ok=True)
    (ROOT/'results/showcase'/('robot_normal_derivatives'+suffix+'.json')).write_text(json.dumps(manifest,indent=2))
    mapping={robot:{} for robot in ['microduck','microduck_roller']}
    for row in manifest:
        robot=Path(row['target']).parent.name
        for geom in row['geometry_ids']:mapping[robot][geom]='res://'+row['target'].removeprefix('godot/')
    (ROOT/'godot/atelier'/('robot_normal_map'+suffix+'.json')).write_text(json.dumps(mapping,indent=2))
    print('Derived',len(manifest),'visual meshes; source assets unchanged',flush=True)


if __name__=='__main__':main()
