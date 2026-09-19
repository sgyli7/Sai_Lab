"""Feed post-action physical states to the unchanged native task metrics."""
import numpy as np
from sim2sim.standalone.sprint import metrics

def score_samples(states,case,commands,selections):
 count=round(case['seconds']/.02)
 if len(states)<count+1:raise ValueError('Post-action scoring requires the final physical state')
 data=np.asarray(states[1:count+1],float)
 if data.shape!=(count,13) or not np.isfinite(data).all():raise ValueError('Invalid states')
 q=data[:,3:7];q=q/np.linalg.norm(q,axis=1,keepdims=True)
 yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
 tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1)))
 v=data[:,10:13];cy,sy=np.cos(yaw),np.sin(yaw)
 velocity=np.stack((cy*v[:,0]+sy*v[:,1],-sy*v[:,0]+cy*v[:,1],v[:,2]),axis=1)
 rows=[dict(xy=d[:2].tolist(),z=float(d[2]),tilt=float(tilt[i]),yaw=float(yaw[i]),vel=velocity[i].tolist()) for i,d in enumerate(data)]
 actions=[dict(t=i*.02,skill='sprint' if selections[i] else 'walking',requested_command=commands[i]) for i in range(count)]
 return metrics(rows,actions,case)
