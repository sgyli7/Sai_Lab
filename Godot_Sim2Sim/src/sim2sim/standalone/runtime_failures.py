"""Generate deliberately invalid policies and verify native failures are explicit."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import re

import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto

from sim2sim.paths import sim2sim_root


def validate(output,executable=None,project=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    def model(name,input_shape=(1,61),output_shape=(1,14),dtype=TensorProto.FLOAT,value=0.):
        array=np.full(output_shape,value,np.float64 if dtype==TensorProto.DOUBLE else np.float32)
        node=helper.make_node('Constant',[],['action'],value=numpy_helper.from_array(array))
        graph=helper.make_graph([node],name,[helper.make_tensor_value_info('obs',dtype,list(input_shape))],
                                [helper.make_tensor_value_info('action',dtype,list(output_shape))])
        data=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)],ir_version=10)
        path=output/(name+'.onnx');onnx.save(data,path);return str(path)
    good=model('good');nonfinite=model('nonfinite',value=float('nan'))
    empty=output/'empty.onnx';empty.write_bytes(b'')
    corrupt=output/'corrupt.onnx';corrupt.write_bytes(b'not an ONNX protobuf')
    failures=[dict(name='empty',path=str(empty)),dict(name='corrupt',path=str(corrupt)),
              dict(name='input_shape',path=model('input_shape',input_shape=(1,60))),
              dict(name='output_shape',path=model('output_shape',output_shape=(1,13))),
              dict(name='external_dtype',path=model('external_dtype',dtype=TensorProto.DOUBLE))]
    fixture=output/'fixtures.json';fixture.write_text(json.dumps(dict(good=good,nonfinite=nonfinite,invalid_loads=failures),indent=2))
    probe=sim2sim_root()/'godot/tests/native_failure_probe.gd'
    if executable:
        # Release templates omit --script. Export a minimal harness PCK and use
        # an exact copy of the tested player binary/libraries in its own folder.
        # Reuse the initialized SDK/project import cache, as in the production
        # package builder. A brand-new minimal project currently crashes this
        # pinned Godot editor during its first GDExtension import/teardown.
        # That failed harness and its GDB trace are retained in research logs.
        project_dir=output/'project'
        subprocess.run(['cp','--reflink=auto','-a',str(Path(project or sim2sim_root()/'godot').resolve()),str(project_dir)],timeout=60,check=True)
        text=probe.read_text().replace('extends SceneTree','extends Node').replace('func _initialize()', 'func _ready()').replace('quit(', 'get_tree().quit(')
        (project_dir/'probe.gd').write_text(text)
        (project_dir/'probe.tscn').write_text('[gd_scene load_steps=2 format=3]\n[ext_resource type="Script" path="res://probe.gd" id="1"]\n[node name="Probe" type="Node"]\nscript = ExtResource("1")\n')
        settings=project_dir/'project.godot'
        settings.write_text(re.sub(r'(?m)^run/main_scene(?:\.standalone)?=.*$',lambda match:match[0].split('=')[0]+'="res://probe.tscn"',settings.read_text()))
        shutil.copyfile(sim2sim_root()/'godot/export_presets.cfg',project_dir/'export_presets.cfg')
        executable=Path(executable).resolve();player=output/'Probe.arm64';shutil.copy2(executable,player)
        for library in executable.parent.glob('*.so*'):shutil.copy2(library,output/library.name)
        with (output/'export.log').open('w') as log:
            subprocess.run(['godot','--headless','--path',str(project_dir),'--export-pack','Linux ARM64 Standalone',str(output/'Probe.pck')],stdout=log,stderr=subprocess.STDOUT,timeout=30,check=True)
        command=[str(player),'--headless']
    else:
        command=['godot','--headless','--path',str(Path(project or sim2sim_root()/'godot').resolve()),
                 '--script','res://tests/native_failure_probe.gd']
    command+=['--','--fixtures='+str(fixture)]
    with (output/'process.log').open('w') as log:
        try:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=60)
            returncode=result.returncode
        except subprocess.TimeoutExpired:
            returncode=124
    lines=[line.removeprefix('NATIVE_FAILURE_RESULT ') for line in (output/'process.log').read_text().splitlines() if line.startswith('NATIVE_FAILURE_RESULT ')]
    report=json.loads(lines[-1]) if lines else dict(passed=False,error='No native failure result')
    report.update(command=command,returncode=returncode)
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    if returncode or not report['passed']:raise RuntimeError('Native failure probe failed; inspect '+str(output))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--executable');p.add_argument('--project');a=p.parse_args()
    print(json.dumps(validate(a.out,a.executable,a.project)))
