from prepare import *
sel=res['selected']
for jaw in [.24,.25,.26,.27,.28,.29]:
 d.qpos[:]=cold;d.qpos[qi]=sel['pre_q'];setjaw(jaw);mj.mj_forward(m,d)
 print('JAW',jaw,[(m.geom(c.geom1).name,m.geom(c.geom2).name,float(c.dist)) for c in d.contact if c.dist<0],flush=True)
