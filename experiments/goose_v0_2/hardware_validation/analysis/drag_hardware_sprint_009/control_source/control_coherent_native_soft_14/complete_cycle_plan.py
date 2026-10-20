"""C2 full backwards reference, no simulator-state API."""
import numpy as np
from scipy.optimize import brentq
F=1.2;V=.15;STRIDE=.125;A=.024;HEIGHT=0.;SWING_HEIGHT=.015;RATIO=.05/.022;D=.525;AX=.2;RAMP=1.875*V/AX;END=D/V+RAMP;PHASE_BRAKE_X=.4;PHASE_BRAKE_T=RAMP+(PHASE_BRAKE_X-.5*V*RAMP)/V;TB=END-PHASE_BRAKE_T
# Four cycles, then planted DS while body finishes braking. Quintic phase is
# monotone and C2; no hidden preparatory interval is excluded from END.
def smooth(u):
 u=np.clip(u,0,1);return 10*u**3-15*u**4+6*u**5

def poly(c,t,n=0):return sum(c[k]*np.prod(np.arange(k-n+1,k+1))*t**(k-n) for k in range(n,len(c)))

def quintic(y,v,a,Y,VV,AA,T):
 M=np.array([[T**3,T**4,T**5],[3*T*T,4*T**3,5*T**4],[6*T,12*T*T,20*T**3]]);return np.r_[y,v,a/2,np.linalg.solve(M,[Y-y-v*T-a*T*T/2,VV-v-a*T,AA-a])]
pc=quintic(PHASE_BRAKE_X/STRIDE,F,0,4.,0,0,TB)
def travel(t):
 t=np.clip(t,0,END)
 def ramp(s):
  u=np.clip(s/RAMP,0,1);return V*RAMP*(2.5*u**4-3*u**5+u**6),V*smooth(u),V/RAMP*(30*u*u-60*u**3+30*u**4)
 if t<RAMP:return ramp(t)
 if t>END-RAMP:
  x,v,a=ramp(END-t);return D-x,v,-a
 return .5*V*RAMP+V*(t-RAMP),V,0.
def phase(t,n=0):
 if t<=PHASE_BRAKE_T:return travel(t)[n]/STRIDE
 if t>=END:return 4. if n==0 else 0.
 return poly(pc,t-PHASE_BRAKE_T,n)
def lateral(t):
 p=phase(t);v=phase(t,1);a=phase(t,2);w=2*np.pi;return -A*np.sin(w*p),-A*w*np.cos(w*p)*v,A*w*w*np.sin(w*p)*v*v-A*w*np.cos(w*p)*a

def height(t):
 # Lower at double support/longest sagittal reach, highest at mid single support.
 # The initial crouch envelope is part of the same backwards-command clock.
 return -HEIGHT*np.cos(2*np.pi*phase(t))**2*smooth(t/.35)
W=[]
for k in range(4):W.extend([('left',.16 if k==0 else k+.01,k+.49),('right',k+.51,k+.99)])
WINDOWS=[dict(side=s,phase_start=a,phase_end=b,start=brentq(lambda t:phase(t)-a,0,END),end=brentq(lambda t:phase(t)-b,0,END)) for s,a,b in W]
def feet_targets(t,coldfeet):
 anchors={s:np.array(p,float).copy() for s,p in coldfeet.items()};active=None
 for w in WINDOWS:
  if t<w['start']:break
  side=w['side'];back,vel,_=travel(w['end']);land=np.array(coldfeet[side],float).copy();land[0]-=back+.5*vel*(1/F-.48/F)
  if w is WINDOWS[1]:land[0]+=.010 # only the first right landing; later formulas unchanged
  if t>=w['end']:anchors[side]=land
  else:
   u=(phase(t)-w['phase_start'])/(w['phase_end']-w['phase_start']);u=float(np.clip(u,0,1));anchors[side]+=smooth(u)*(land-anchors[side]);anchors[side][2]+=SWING_HEIGHT*64*u**3*(1-u)**3;active=side;break
 return anchors,active
PARAMETERS=dict(frequency_cruise_Hz=F,cruise_m_s=V,steady_samefoot_stride_m=STRIDE,lateral_COM_amplitude_m=A,bodyZ_peak_to_peak_m=HEIGHT,bodyZ_phase='lowDS/highSS',initial_crouch_C2_s=.35,foot_lift_m=SWING_HEIGHT,swing_phase_fraction=.48,first_left_phase_start=.16,first_left_phase_end=.49,first_right_landing_forward_correction_m=.010,body_acceleration_peak=.2,speed_ramp_s=RAMP,planned_distance_m=D,command_to_stop_s=END,phase_brake_start_s=PHASE_BRAKE_T,phase_brake_end_s=END,planned_cycles=4,planned_alternating_swings=8,windows=WINDOWS,actual_steps_claimed=0,qualification='onlyfutureactualGoose/cart/cargo>=.5m,>=.1m/s,stop<=5s+hold3s counts; requestsnotsteps',hidden_time_excluded_s=0)


def support_force_objective(t,measured_total_normal_N):
 """Soft phase objective, not a hard Fn0 or invented available mouth force.
 Each .48-cycle window unloads its own side smoothly; actual source contacts
 and PD remain authoritative. Adjacent DS is .02cycle=16.667ms at cruise.
 """
 for w in WINDOWS:
  if w['start']<=t<=w['end']:
   u=(phase(t)-w['phase_start'])/(w['phase_end']-w['phase_start'])
   # C2 unload then reload, never force early/late small-clearance Fn0.
   envelope=smooth(min(u/.2,1))*smooth(min((1-u)/.2,1))
   return dict(unload_side=w['side'],normal_soft_target_N=.5*measured_total_normal_N*(1-envelope),weight_blend=envelope,force_is_soft=True)
 return dict(unload_side='left',normal_soft_target_N=.5*measured_total_normal_N,weight_blend=0.,force_is_soft=True)
