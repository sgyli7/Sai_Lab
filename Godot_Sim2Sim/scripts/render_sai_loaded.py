"""Render recorded MuJoCo physical states; this does not simulate or alter the replay."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from sai_loaded_mujoco import LoadedTerrain


def main():
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.episode/'report.json').read_text());rows=json.loads((a.episode/'trace.json').read_text())
    model=LoadedTerrain(report['seed'],report['kind'],report['mass'],report.get('tread',.18),report.get('terrain_scale',1.)).model()
    data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=480,width=720);camera=mujoco.MjvCamera()
    camera.distance=1.12;camera.azimuth=135;camera.elevation=-23
    option=mujoco.MjvOption();option.geomgroup[:]=1;model.geom_rgba[model.geom('payload').id]=[.95,.65,.10,1.]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    command=[shutil.which('ffmpeg'),'-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size','720x480','-framerate','25','-i','-',
             '-c:v','libx264','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out)]
    child=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for row in rows[::2]:
            if 'qpos' not in row:raise ValueError('This diagnostic trace predates joint-state recording')
            data.qpos[:]=row['qpos'];mujoco.mj_forward(model,data)
            camera.lookat[:]=data.qpos[:3];camera.lookat[2]+=.02
            renderer.update_scene(data,camera=camera,scene_option=option);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,720,40),fill=(20,25,30))
            label=('Trained suspension' if report.get('parameters') else 'Baseline')+' | MuJoCo | '+report['kind']+' | '+str(round(report['mass']*1000))+' g cargo | t='+str(round(row['time'],2))+' s'
            draw.text((12,12),label,fill='white');child.stdin.write(np.asarray(im).tobytes())
        child.stdin.close();code=child.wait(timeout=30)
        if code:raise RuntimeError('Video encoding failed')
    finally:
        renderer.close()
        if child.poll() is None:child.terminate();child.wait(timeout=5)

if __name__=='__main__':main()
