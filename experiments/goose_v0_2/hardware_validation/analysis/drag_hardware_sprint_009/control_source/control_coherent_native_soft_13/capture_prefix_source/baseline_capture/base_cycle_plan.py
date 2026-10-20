"""One finite coordinated reference, zero dynamics. No online state-writing API."""
import numpy as np
from scipy.optimize import brentq
F=.8;A=.040;V=.14;H=.010;SWING=.40;PERIOD=1.25;MID_FOOT_X=.015;RATIO=.05/.022;D=.5;AX=.2;RAMP=1.875*V/AX;TRANSPORT=D/V+RAMP
wold=2*np.pi*1.5;oldentry=1.5*np.pi-wold*.19/2;Told=.5
mat=np.array([[Told**3,Told**4,Told**5],[3*Told**2,4*Told**3,5*Told**4],[6*Told,12*Told**2,20*Told**3]])
c=np.r_[0.,0.,0.,np.linalg.solve(mat,[.022*np.sin(oldentry),.022*wold*np.cos(oldentry),-.022*wold*wold*np.sin(oldentry)])]
def poly(c,t,n=0):return sum(c[k]*np.prod(np.arange(k-n+1,k+1))*t**(k-n) for k in range(n,len(c)))
y0=poly(c,.5);v0=poly(c,.5,1)/2;a0=poly(c,.5,2)/4
PHI0=np.pi+np.arcsin(-y0/A);OM0=v0/(A*np.cos(PHI0));AL0=(a0/A+OM0**2*np.sin(PHI0))/np.cos(PHI0);TR=.5;OM=2*np.pi*F;PHI1=PHI0+.5*(OM0+OM)*TR
mm=np.array([[TR**3,TR**4,TR**5],[3*TR**2,4*TR**3,5*TR**4],[6*TR,12*TR**2,20*TR**3]])
pc=np.r_[PHI0,OM0,.5*AL0,np.linalg.solve(mm,[PHI1-PHI0-OM0*TR-.5*AL0*TR**2,OM-OM0-AL0*TR,-AL0])]
def phase(t,n=0):
 s=max(0.,t-1.)
 if s<TR:return poly(pc,s,n)
 return PHI1+OM*(s-TR) if n==0 else OM if n==1 else 0.
def periodic_lateral(t):
 p=phase(t);om=phase(t,1);al=phase(t,2);return A*np.sin(p),A*om*np.cos(p),A*(al*np.cos(p)-om*om*np.sin(p))
CENTERS=[brentq(lambda t:phase(t)-(1.5*np.pi+k*np.pi),1.,10.) for k in range(8)];STARTS=np.array(CENTERS)-SWING/2;ENDS=np.array(CENTERS)+SWING/2;XSTART=float(STARTS[0]);XEND=XSTART+TRANSPORT;SETTLE=float(ENDS[-1]);TS=XEND-SETTLE;ys,vs,acs=periodic_lateral(SETTLE)
sm=np.array([[TS**3,TS**4,TS**5],[3*TS**2,4*TS**3,5*TS**4],[6*TS,12*TS**2,20*TS**3]])
sc=np.r_[ys,vs,.5*acs,np.linalg.solve(sm,[-ys-vs*TS-.5*acs*TS**2,-vs-acs*TS,-acs])]
def lateral(t):
 if t<1.:return poly(c,t/2),poly(c,t/2,1)/2,poly(c,t/2,2)/4
 if t>=SETTLE:
  if t>=XEND:return 0.,0.,0.
  return tuple(poly(sc,t-SETTLE,k) for k in range(3))
 return periodic_lateral(t)
def travel(t):
 s=np.clip(t-XSTART,0,TRANSPORT)
 def ramp(s):
  u=np.clip(s/RAMP,0,1);return V*RAMP*(2.5*u**4-3*u**5+u**6),V*(10*u**3-15*u**4+6*u**5),V/RAMP*(30*u**2-60*u**3+30*u**4)
 if s<RAMP:return ramp(s)
 if s>TRANSPORT-RAMP:
  z,v,a=ramp(TRANSPORT-s);return D-z,v,-a
 return .5*V*RAMP+V*(s-RAMP),V,0.
def feet_targets(t,coldfeet):
 anchors={s:np.array(v,float).copy() for s,v in coldfeet.items()};active=None
 for k,(start,end) in enumerate(zip(STARTS,ENDS)):
  if t<start:break
  side='left' if k%2==0 else 'right';b,v,a=travel(end);land=np.array(coldfeet[side],float).copy();land[0]=-b+MID_FOOT_X-.5*v*(PERIOD-SWING)
  if t>=end:anchors[side]=land
  else:
   active=side;u=(t-start)/SWING;s=10*u**3-15*u**4+6*u**5;anchors[side]=anchors[side]+(land-anchors[side])*s;anchors[side][2]+=H*64*u**3*(1-u)**3;break
 return anchors,active
PARAMETERS={'frequency_Hz':F,'lateral_COM_amplitude_m':A,'body_roll_amplitude_rad':RATIO*A,'cruise_body_backward_m_s':V,'body_peak_accel_m_s2':AX,'body_speed_ramp_s':RAMP,'transport_distance_m':D,'transport_duration_s':TRANSPORT,'first_swing_duration_s':SWING,'swing_height_m':H,'steady_per_foot_cycle_stride_m':V/F,'initial_DS_prep_s':1.,'phase_speed_transition_s':TR,'phase_initial_phi_omega_alpha':[PHI0,OM0,AL0],'phase_steady_omega':OM,'first_Xstart_s_from_motion':XSTART,'Xstop_s_from_motion':XEND,'final_DS_lateral_settle_start_s':SETTLE,'final_DS_lateral_settle_duration_s':TS,'swing_start_s':STARTS.tolist(),'swing_end_s':ENDS.tolist(),'planned_swing_sides':['left' if k%2==0 else 'right' for k in range(8)],'count_semantics':'8requested alternating swings=4requested LRpairs,notrealsteps;actualair>=1mm50ms+backlanding+stanceeventrequired;4cyclesnotaprimaryhardgateforfirstphysics','actual_live_state_write_API':False}
