import base_cycle_plan as base
from base_cycle_plan import *
STARTS=np.array([base.STARTS[0]]);ENDS=np.array([base.ENDS[0]]);END=float(ENDS[0]);FOLLOW=.8;XEND=END+FOLLOW;cy0,cv0,ca0=base.lateral(END);bx,bv,ba=base.travel(END)
def quintic(y,v,a,Y,T):
 M=np.array([[T**3,T**4,T**5],[3*T**2,4*T**3,5*T**4],[6*T,12*T**2,20*T**3]]);return np.r_[y,v,.5*a,np.linalg.solve(M,[Y-y-v*T-.5*a*T*T,-v-a*T,-a])]
lat=quintic(cy0,cv0,ca0,0.,FOLLOW);STOP=.4;BFINAL=bx+.5*bv*STOP;back=quintic(bx,bv,ba,BFINAL,STOP)
def lateral(t):
 if t<=END:return base.lateral(t)
 if t>=XEND:return 0.,0.,0.
 return tuple(poly(lat,t-END,n) for n in range(3))
def travel(t):
 if t<=END:return base.travel(t)
 if t>=END+STOP:return BFINAL,0.,0.
 return tuple(poly(back,t-END,n) for n in range(3))
def feet_targets(t,coldfeet):return base.feet_targets(min(t,END),coldfeet)
PARAMETERS=dict(base.PARAMETERS,scope='FIRST LEFT MECHANISM ONLY, laterinvalidcycleexplicitlyexcluded',swing_start_s=STARTS.tolist(),swing_end_s=ENDS.tolist(),planned_swing_sides=['left'],planned_transport_distance_m=None,maximum_diagnostic_body_back_m=BFINAL,followthrough_s=FOLLOW,body_stop_tail_s=STOP,Xstop_s_from_motion=END+STOP,reference_end_s_from_motion=XEND,count_semantics='0actualstepsclaimed;onlyactualleftair>=1mm50ms+backlandingcancompleteexperiment')
