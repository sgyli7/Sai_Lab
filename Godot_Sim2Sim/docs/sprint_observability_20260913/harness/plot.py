from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('results/sprint_observability_20260913')
ARMS=['immediate','state','contacts','history','duplicate_current']
LABELS=['Immediate','+ body state','+ contacts','+ history','Repeat current\n(no new information)']
COLORS=['#64748b','#2563eb','#d97706','#16a34a','#9333ea']
fig, axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained',sharey=True)
for ax,experiment,title in zip(axes,['predictor','predictor_long'],['40 fixed epochs','640 fixed epochs']):
    results=json.loads((ROOT/experiment/'analysis.json').read_text())['results']
    for i,(arm,color) in enumerate(zip(ARMS,COLORS)):
        for j,subset in enumerate(['moving','transition']):
            values=[results[f'{arm}_{seed}'][subset]['normalized_rmse'] for seed in [997101,997102]]
            x=j+(i-2)*.16
            ax.bar(x,np.mean(values),width=.145,color=color,label=LABELS[i] if j==0 else None,alpha=.8)
            ax.scatter([x]*2,values,s=14,color='black',zorder=3)
    ax.set_xticks([0,1],['Moving','First second after command change'])
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Normalized response RMSE (lower is better)')
handles,labels=axes[1].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=5,frameon=False,fontsize=9)
fig.suptitle('Native Jolt information diagnostic — two learner seeds, reset-seed-isolated development test')
fig.savefig(ROOT/'information_ablation.png',dpi=170)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,seed in zip(axes,[997101,997102]):
    for arm,label,color in zip(ARMS,LABELS,COLORS):
        d=json.loads((ROOT/'predictor_long/learners'/f'{arm}_{seed}'/'completed.json').read_text())
        curves=d['curves']
        ax.plot([x['epoch'] for x in curves],[x['validation_mse'] for x in curves],label=label,color=color)
    ax.set_title(f'Learner seed {seed}')
    ax.set_xlabel('Epoch (endpoint fixed at 640)')
    ax.set_ylabel('Validation normalized MSE')
    ax.set_yscale('log')
    ax.grid(alpha=.2)
handles,labels=axes[1].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=5,frameon=False,fontsize=9)
fig.savefig(ROOT/'learning_curves.png',dpi=170)
