"""Native simulator frames, sampled inside normal control steps."""
import argparse,json,os
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .tasks import TASKS,SESSION,DT
from .models import NativeAnchor
from .world import World
from .evaluate import record,summarize

def capture(skill,source,backend,out,seed=100,condition="default",label=None,entry="reset",scene_robot=None):
    task=TASKS[skill];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if scene_robot is not None:task=replace(task,robot=scene_robot)
    os.environ.setdefault("SIM2SIM_FORCE_GL","1")
    os.environ.setdefault("SIM2SIM_DISPLAY_DRIVER","x11")
    policy=NativeAnchor(source);w=World(task,backend,headless=backend!="godot",time_input_s=policy.time_input_s,heading_input=policy.heading_input,yaw_memory_input=policy.yaw_memory_input)
    renderer=None;frames=[];rows=[]
    try:
        obs=w.reset(seed,condition)
        if backend=="godot":
            # Reset is still frozen; zooming here does not advance dynamics.
            w.backend._client.call({"cmd":"camera_zoom","d":-10.})
        if entry=="standing":obs=w.enter_from_standing()
        if backend=="mujoco":
            import mujoco
            w.mj.model.vis.global_.offwidth=900;w.mj.model.vis.global_.offheight=540
            renderer=mujoco.Renderer(w.mj.model,height=540,width=900)
            camera=mujoco.MjvCamera();camera.distance=.65;camera.elevation=-15;camera.azimuth=np.degrees(np.arctan2(w.heading[1],w.heading[0]))+70
        for k in range(round(task.seconds/DT)):
            a=policy(obs[None])[0];path=out/f"frame_{k:04d}.png" if k%4==0 else None
            w.send(a,capture_path=None if path is None or backend!="godot" else str(path.resolve()))
            obs=w.recv();rows.append(record(w,a))
            if path is not None:
                if backend=="mujoco":
                    camera.lookat[:]=w.state.base_pos+[0,0,.015]
                    renderer.update_scene(w.mj.data,camera=camera)
                    Image.fromarray(renderer.render()).save(path)
                if path.exists():
                    frame=Image.open(path).convert("RGB");frame.thumbnail((900,540))
                    framed=Image.new("RGB",(900,585),(20,25,35));framed.paste(frame,((900-frame.width)//2,45+(540-frame.height)//2))
                    draw=ImageDraw.Draw(framed)
                    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",19)
                    draw.text((16,12),f"{label or skill} | {backend} | t={(k+1)*DT:.2f}s",font=font,fill="white")
                    frames.append(framed)
        result=summarize(task,rows,w.heading);result.update(backend=backend,seed=seed,condition=condition,entry=entry,policy=str(source),physics=w.physics)
        (out/"metrics.json").write_text(json.dumps(result,indent=2))
        frames[0].save(out/"clip.gif",save_all=True,append_images=frames[1:],duration=80,loop=0,optimize=False)
        frames[min(len(frames)-1,12)].save(out/"preview.png")
    finally:
        if renderer is not None:renderer.close()
        w.close()
    return out/"clip.gif"

def main():
    p=argparse.ArgumentParser();p.add_argument("--skill",required=True);p.add_argument("--source",type=Path)
    p.add_argument("--backend",default="godot");p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=100);p.add_argument("--condition",default="default");p.add_argument("--label")
    p.add_argument("--entry",choices=("reset","standing"),default="reset")
    p.add_argument("--scene-robot")
    a=p.parse_args();print(capture(a.skill,a.source or TASKS[a.skill].source,a.backend,a.out,a.seed,a.condition,a.label,a.entry,a.scene_robot))

if __name__=="__main__":main()
