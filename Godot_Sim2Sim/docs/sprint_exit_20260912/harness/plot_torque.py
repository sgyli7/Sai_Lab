from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('results/sprint_exit_20260912');p=json.loads((r/'bam_source/bam/params/xl330/m6.json').read_text())
kt=p['kt'];res=p['R'];limit=kt*1.75;dq=np.linspace(0,22,400)
fig,ax=plt.subplots(figsize=(8,5.1))
fig.subplots_adjust(left=.11,right=.98,top=.91,bottom=.24)
ax.plot(dq,limit-.053*dq,label='Frozen Jolt: clamped position drive − viscous term',lw=2.5)
ax.plot(dq,np.minimum(limit,kt*7.5/res-kt*kt/res*dq)-p['friction_viscous']*dq,label='Source BAM: current / voltage envelope (7.5 V)',lw=2.5)
ax.axhline(0,color='#777777',lw=.8);ax.set(xlabel='Joint angular velocity (rad/s)',ylabel='Maximum positive-command torque (N·m)',title='Same current rating, different attainable torque at speed',ylim=(-.3,.7),xlim=(0,22))
ax.grid(alpha=.2);ax.legend(loc='lower left',fontsize=9)
fig.text(.02,.045,'Electrical + viscous terms only; excludes dry/load friction, delays, supply sag and contacts.\nAnalytical counterfactual, not a measured motor curve or proof of the cause of falling.',fontsize=9)
fig.savefig(r/'torque_comparison/envelope.png',dpi=170,bbox_inches='tight')
