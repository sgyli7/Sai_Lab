from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=Path('results/sprint_gpu_match_20260912');d=Path('docs/sprint_gpu_match_20260912');s=json.loads((r/'native_comparison.json').read_text());names=['native_s05','native_source_graph','native_jolt_graph','native_positive','native_cat'];labels=['S05\nbaseline','Source-foot\nGPU','Game-foot\nGPU','Positive\ncontrol','CaT'];colors=['#64748b','#e6a05c','#e6a05c','#e6a05c','#2a9081'];x=np.arange(5)
passes=[s[n]['completed']['candidate_pass'] for n in names];speeds=[next(c['candidate_vx'] for c in s[n]['completed']['comparisons'] if c['case']=='sprint_long') for n in names]
fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
axes[0].bar(x,speeds,color=colors,width=.65);axes[0].set_ylim(0,.34);axes[0].set_ylabel('Native long-course speed (m/s)')
for i,y in enumerate(speeds):axes[0].text(i,y+.008,f'{y:.3f}',ha='center',fontsize=10)
axes[1].bar(x,passes,color=colors,width=.65);axes[1].set_ylim(0,142);axes[1].set_ylabel('Native full cases passed / 128');axes[1].axhline(128,color='#994b4b',ls='--',lw=1)
for i,y in enumerate(passes):axes[1].text(i,y+2,str(y),ha='center',fontsize=10)
for ax in axes:
 ax.set_xticks(x,labels);ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
fig.suptitle('GPU sprint learning: speed alone is not enough',fontsize=15)
fig.supxlabel('Development seeds only. All zero falls; CaT loses 6 baseline successes. No promotion.',fontsize=9)
fig.savefig(d/'native_comparison.png',dpi=180);fig.savefig(d/'native_comparison.svg');plt.close(fig)
vector=d/'native_comparison.svg';vector.write_text('\n'.join(line.rstrip() for line in vector.read_text().splitlines())+'\n')
print(d/'native_comparison.png')
