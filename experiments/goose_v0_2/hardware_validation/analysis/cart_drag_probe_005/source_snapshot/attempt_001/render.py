import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='osmesa';os.environ['PYOPENGL_PLATFORM']='osmesa';os.environ['LIBGL_ALWAYS_SOFTWARE']='true';os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']='/usr/share/glvnd/egl_vendor.d/50_mesa.json';os.environ['EGL_PLATFORM']='surfaceless'
os.environ['LP_NUM_THREADS']='1';os.environ['GALLIUM_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,mujoco as mj,subprocess,json
R=Path(__file__).parent;m=mj.MjModel.from_binary_path(str(R/'actual_scene.mjb'));d=mj.MjData(m);t=np.load(R/'trajectory.npz');r=mj.Renderer(m,height=480,width=640);from OpenGL import GL
print('CPU_RENDERER',GL.glGetString(GL.GL_RENDERER),flush=True)
cam=mj.MjvCamera();cam.lookat[:]=[.5,0,.25];cam.distance=1.8;cam.azimuth=105;cam.elevation=-12
p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','640x480','-r','20','-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-pix_fmt','yuv420p',str(R/'continuous_original_speed.mp4')],stdin=subprocess.PIPE)
for idx in range(0,len(t['time']),10):
 d.qpos[:]=t['qpos'][idx];d.qvel[:]=t['qvel'][idx];mj.mj_forward(m,d);r.update_scene(d,camera=cam);p.stdin.write(r.render().tobytes())
p.stdin.close();p.wait();r.close();print('VIDEO',p.returncode,flush=True)
