from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('results/sprint_target_gpu_20260912');d=Path('docs/sprint_target_gpu_20260912')
rows=json.loads((r/'noise_probe/completed.json').read_text())['rows']
fig,axes=plt.subplots(1,3,figsize=(10.5,3.4),sharey=True)
for ax,seed in zip(axes,[927001,927007,927008]):
 for j,sigma in enumerate([0.,.005,.02]):
  values=[x['signed_yaw_rate'] for x in rows if x['seed']==seed and x['sigma']==sigma]
  ax.scatter(j+np.linspace(-.08,.08,len(values)),values,s=32,color='#2463a5',zorder=3)
  ax.hlines(np.mean(values),j-.16,j+.16,color='#163e68',linewidth=2)
 ax.axhline(.48,color='#b85538',linestyle='--',linewidth=1.2)
 ax.set_xticks([0,1,2],['0','.005','.02']);ax.set_xlabel('Action noise sigma (rad)')
 ax.set_title('Reset seed '+str(seed),fontsize=10);ax.set_ylim(.2,.85);ax.grid(axis='y',alpha=.2)
 ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Signed yaw response (rad/s)')
fig.suptitle('Frozen policy: exploration changes the trajectory',fontsize=13)
fig.tight_layout();fig.savefig(d/'noise_diagnostic.png',dpi=160);plt.close(fig)
