import numpy as np
def smooth(u):
 u=float(np.clip(u,0,1));return 10*u**3-15*u**4+6*u**5
def tcp(cold,t):return np.asarray(cold)+np.array([.000347296355334,0,.0019696155060244])*(1-smooth(t/.2))
def jaw(t):return .29*(1-smooth((t-.2)/.6))
def blend(t):return smooth((t-.2)/.6)
