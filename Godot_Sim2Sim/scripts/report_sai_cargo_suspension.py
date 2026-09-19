"""Plot paired measured payload acceleration; preserve all individual case ratios."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/sai-cargo-suspension-20260915'
OUT=ROOT/'docs/sai-suspension-20260915'

def main():
    OUT.mkdir(exist_ok=True);summary={};fig,axes=plt.subplots(2,2,figsize=(12,7.2),layout='constrained')
    for ax,engine,name in zip(axes[0],['MuJoCo 1 kHz','Jolt 2 kHz'],['accept-mujoco','accept-jolt']):
        data=json.loads((RESULTS/name/'acceptance.json').read_text());summary[name]=dict(passed=data['passed'],pairs=len(data['pairs']),groups={})
        for j,group in enumerate(['continuous','stairs']):
            rows=[p for p in data['pairs'] if p['case']['kind'].startswith(('up','down','mixed'))==(group=='stairs')]
            values=np.array([p['ratios']['cargo_accel_rms'] for p in rows]);summary[name]['groups'][group]={k:float(np.mean([r['ratios'][k] for r in rows])) for k in rows[0]['ratios']}
            ax.bar(j-.16,1.,.3,color='#a7adb5',label='Baseline' if j==0 else None)
            ax.bar(j+.16,values.mean(),.3,color='#178a8e',label='Trained' if j==0 else None)
            ax.scatter(j+.16+np.linspace(-.08,.08,len(values)),values,s=12,alpha=.55,color='#1c424b',zorder=3)
            ax.text(j+.16,values.mean()+.07,f'{(1-values.mean())*100:.1f}% lower',ha='center',fontsize=10)
        ax.set(xticks=[0,1],xticklabels=['Uneven terrain','Stairs / mixed'],ylabel='Cargo acceleration RMS / paired baseline',title=engine,ylim=(0,1.45));ax.legend(frameon=False,loc='upper right');ax.grid(axis='y',alpha=.15)
    for ax,case,title in zip(axes[1],['02-washboard','10-down40'],['Washboard, 50 g free cargo','40 mm descent, 50 g free cargo']):
        for label,color,name in [('Baseline','#8e949d','baseline'),('Trained','#178a8e','candidate')]:
            samples=np.load(RESULTS/'accept-mujoco'/f'{case}-{name}'/'physics.npz')['samples']
            ax.plot(samples[:,0],np.linalg.norm(samples[:,4:7],axis=1),color=color,label=label,lw=.85)
        ax.set(title=title,xlabel='Simulation time (s)',ylabel='Cargo acceleration (m/s²)');ax.grid(alpha=.15);ax.legend(frameon=False)
    fig.suptitle('Measured cargo comfort in full articulated physics',fontsize=16)
    fig.savefig(OUT/'cargo-comparison.png',dpi=160);plt.close(fig)
    (OUT/'cargo-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
