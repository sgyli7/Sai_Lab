"""Package nine complete raw videos and telemetry without redistributing weights."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
output=ROOT/'results/showcase/MicroDuck-Service-Bay-raw-evidence.zip'
index={}
with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as archive:
    archive.writestr('README.txt', '''Nine real ONNX-controlled Godot/Jolt demonstrations, seed 61000.
Each raw.mp4 contains the initial policy handoff plus the complete action.
Videos preserve recorded wall-clock timestamps; no speed-up or recovery reset.
The frame entries in capture.json describe source JPEG timestamps. Individual
JPEGs are not included; their encoded video is raw.mp4. trajectory.npz contains
all control-step samples. See capture.json for the selected model SHA-256.
Roller push/coast/brake failed its braking criterion; do not describe it as a pass.
The public source repository contains the workshop and current control runtime.
No robot meshes, fonts, ONNX files, training checkpoints or credentials included.
''')
    for directory in sorted((ROOT/'results/showcase').glob('04_*')):
        if not (directory/'capture.json').exists():continue
        skill=directory.name.removeprefix('04_')
        index[skill]={}
        for name in ['raw.mp4','capture.json','trajectory.npz','source.json','godot.log','resource_before.json']:
            p=directory/name
            data=p.read_bytes()
            if name.endswith(('.json','.log')):
                data=data.decode().replace(str(ROOT)+'/', '').encode()
            archive.writestr(f'{skill}/{name}',data,compress_type=zipfile.ZIP_STORED if name.endswith('.mp4') else zipfile.ZIP_DEFLATED)
            index[skill][name]=hashlib.sha256(data).hexdigest()
    archive.writestr('checksums.json',json.dumps(index,indent=2)+'\n')
    for p in [ROOT/'docs/showcase_physics_equivalence.json',ROOT/'docs/showcase_runs.json',ROOT/'docs/media/edit.json',ROOT/'docs/media/manifest.json']:
        archive.write(p,p.name)
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
assert len(index)==9
print(output.name,output.stat().st_size,hashlib.sha256(output.read_bytes()).hexdigest())
