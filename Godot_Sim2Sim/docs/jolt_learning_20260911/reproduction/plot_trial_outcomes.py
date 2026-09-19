"""Plot only completed physical endpoints; each point is one training seed."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p=Path("results/jolt_learning_20260911")
rows=json.loads((p/"trial_analysis.json").read_text())["trials"]
families=[("r0_legacy",False,"Legacy / 0.262M"),("r0_command_heading_v1",False,"Aligned heading / 0.262M"),("r1_masked",False,"Task state masked / 0.262M"),("r1_full",False,"Task state full / 0.262M"),("r2_sagittal",False,"Restricted joints / 0.262M"),("r1_full",True,"Task state full / 1.049M"),("r2_sagittal",True,"Restricted joints / 1.049M"),("r3_online_kl",True,"Online teacher KL / 1.049M"),("r3_replay_kl",True,"Success replay KL / 1.049M")]
fig,ax=plt.subplots(figsize=(10.5,6.3),layout="constrained")
colors={71:"#2563eb",72:"#e8590c",73:"#16a085"}
for y,(prefix,extension,label) in enumerate(families):
 group=[r for r in rows if r.get("completed") and r["name"].startswith(prefix+"_s") and r["name"].endswith("_1m")==extension]
 for r in group:
  ax.scatter(r["passes"],y+(r["seed"]-72)*.14,color=colors[r["seed"]],s=55,zorder=3)
  ax.annotate(str(r["passes"]),(r["passes"],y+(r["seed"]-72)*.14),xytext=(5,0),textcoords="offset points",va="center",fontsize=9)
ax.axvline(51,color="#475569",ls="--",label="Frozen baseline: 51 / 56")
ax.set(yticks=range(len(families)),yticklabels=[x[2] for x in families],xlim=(-1,57),xlabel="Passed active-braking episodes out of 56",title="Fixed Jolt: paired development outcomes")
ax.invert_yaxis();ax.grid(axis="x",alpha=.2);ax.spines[["top","right"]].set_visible(False)
for seed,color in colors.items():ax.scatter([],[],color=color,label=f"Training seed {seed}")
ax.legend(loc="upper left",framealpha=.95)
fig.savefig(p/"trial_outcomes.png",dpi=160)
