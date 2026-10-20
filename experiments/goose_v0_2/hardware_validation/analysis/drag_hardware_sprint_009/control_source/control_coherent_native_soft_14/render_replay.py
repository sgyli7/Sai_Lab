import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='osmesa';os.environ['PYOPENGL_PLATFORM']='osmesa';os.environ['LIBGL_ALWAYS_SOFTWARE']='true';os.environ['GALLIUM_DRIVER']='softpipe';os.environ['LP_NUM_THREADS']='1';os.environ['GALLIUM_NUM_THREADS']='1';os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:3])
from pathlib import Path
import numpy as np,mujoco as mj,subprocess
R=Path(__file__).parent
for folder in sorted(R.glob('attempt_*')):
 m=mj.MjModel.from_binary_path(str(folder/'actual_scene.mjb'));m.vis.headlight.active=1;m.vis.headlight.ambient[:]=.3;m.vis.headlight.diffuse[:]=.4;m.vis.headlight.specular[:]=.05;d=mj.MjData(m);tr=np.load(folder/'trajectory.npz');renderer=mj.Renderer(m,height=480,width=640);cam=mj.MjvCamera();cam.lookat[:]=[.50,0,.27];cam.distance=2.35;cam.azimuth=105;cam.elevation=-12
 p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','640x480','-r','20','-i','-','-an','-c:v','libx264','-threads','1','-preset','fast','-pix_fmt','yuv420p',str(folder/'neutral_proxy_original_speed.mp4')],stdin=subprocess.PIPE)
 for tt in np.arange(0,float(tr['time'][-1])-1e-8,.05):
  idx=int(np.argmin(abs(tr['time']-tt)))
  d.qpos[:]=tr['qpos'][idx];d.qvel[:]=tr['qvel'][idx];mj.mj_forward(m,d);renderer.update_scene(d,camera=cam);p.stdin.write(renderer.render().tobytes())
 p.stdin.close();p.wait();renderer.close();assert p.returncode==0;print('VIDEO_COMPLETE',folder.name,flush=True)
