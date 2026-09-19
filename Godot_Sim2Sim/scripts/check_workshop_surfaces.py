"""Find differently pigmented, same-facing triangles that overlap on one plane.

Input is emitted from real procedural meshes by workshop_surface_probe.gd.
Clockwise downward faces buried at/below the floor are excluded by that probe.
"""
import json,sys,collections
faces=json.load(open(sys.argv[1])); groups=collections.defaultdict(list)
for f in faces:groups[(f['axis'],f['sign'],round(f['plane'],6))].append(f)
def intersect(subject,clip):
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    sign=1 if cross(*clip)>0 else -1
    out=subject
    for a,b in zip(clip,clip[1:]+clip[:1]):
        inp=out;out=[]
        if not inp:break
        prev=inp[-1];dp=cross(a,b,prev)*sign
        for cur in inp:
            dc=cross(a,b,cur)*sign
            if (dc>=0)!=(dp>=0):
                t=dp/(dp-dc);out.append([prev[j]+t*(cur[j]-prev[j]) for j in range(2)])
            if dc>=0:out.append(cur)
            prev=cur;dp=dc
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(out,out[1:]+out[:1])))*.5 if out else 0
bad=[]
for key,fs in groups.items():
    for i,a in enumerate(fs):
        for b in fs[i+1:]:
            if a['color']==b['color']:continue
            aa=[[v[j] for j in range(3) if j!=key[0]] for v in a['vertices']]
            bb=[[v[j] for j in range(3) if j!=key[0]] for v in b['vertices']]
            if any(max(v[j] for v in aa)<=min(v[j] for v in bb) or max(v[j] for v in bb)<=min(v[j] for v in aa) for j in range(2)):continue
            area=intersect(aa,bb)
            if area>1e-7:bad.append(dict(plane=key,colors=[a['color'],b['color']],area=area,center=[sum(v[j] for v in a['vertices'])/3 for j in range(3)]))
if bad:print(json.dumps(sorted(bad,key=lambda x:-x['area'])[:20],indent=2))
print('FAIL' if bad else 'PASS',len(bad),'coplanar triangle overlaps between different pigments')
sys.exit(bool(bad))
