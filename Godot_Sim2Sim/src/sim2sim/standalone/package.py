"""Export an auditable, relocatable ARM64 package from prepared policy resources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from sim2sim.paths import sim2sim_root, load_robot_json
from sim2sim.research.queue import atomic_json


def sha256(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def source_hashes(root,project=None):
    # Include uncommitted source too: a base Git commit alone does not identify a build.
    candidates=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],
                                       cwd=root).decode().split('\0')
    if project:
        # A frozen project can predate files added in another workspace task.
        # Inventory the build's actual Godot tree, including snapshot-only inputs.
        project=Path(project)
        candidates=[name for name in candidates if not name.startswith('godot/')]
        candidates.extend('godot/'+p.relative_to(project).as_posix()
                          for p in project.rglob('*') if p.is_file()
                          and '.godot' not in p.relative_to(project).parts)
    def source(name):
        return Path(project)/name.removeprefix('godot/') if project and name.startswith('godot/') else root/name
    return {name:sha256(source(name)) for name in sorted(set(candidates))
            if name and name.startswith(('godot/','native/','src/')) and source(name).is_file()}


def package(output,archive=False,project=None):
    root=sim2sim_root();output=Path(output).resolve()
    project=(Path(project) if project else root/'godot').resolve()
    if output.exists():raise FileExistsError('Export must use a new directory: '+str(output))
    dependencies=json.loads((root/'native/dependencies.json').read_text())
    version=subprocess.check_output(['godot','--version'],text=True).strip()
    if not version.startswith(dependencies['godot_version']+'.'):
        raise RuntimeError('Editor and pinned export templates differ: '+version)
    deployment=json.loads((project/'runtime_assets/deployment.json').read_text())
    fixture=json.loads((project/'runtime_assets/self_test.json').read_text())
    for skill,item in deployment['policies'].items():
        path=project/item['path'].removeprefix('res://')
        if sha256(path)!=item['sha256']:raise RuntimeError('Prepared model checksum mismatch: '+skill)
    if {x['skill']:x['sha256'] for x in fixture['cases']}!={k:v['sha256'] for k,v in deployment['policies'].items()}:
        raise RuntimeError('Self-test fixtures do not match the packaged models')
    font=deployment.get('ui_font')
    if not font:raise RuntimeError('Prepared standalone interface font is required')
    font_path=project/font['path'].removeprefix('res://')
    if sha256(font_path)!=font['sha256']:raise RuntimeError('Prepared interface font checksum mismatch')
    output.mkdir(parents=True)
    binary=output/'MicroDuck.arm64'
    with (output/'export.log').open('w') as log:
        subprocess.run(['godot','--headless','--path',str(project),'--export-release',
                        'Linux ARM64 Standalone',str(binary)],stdout=log,stderr=subprocess.STDOUT,
                       timeout=180,check=True)
    required=['MicroDuck.arm64','MicroDuck.pck','libmicroduck_policy.linux.release.arm64.so','libonnxruntime.so.1']
    for name in required:
        if not (output/name).is_file():raise RuntimeError('Export did not produce '+name)
    # Editor/debug success does not prove the release extension supports this
    # model contract. Exercise the actual exported binary before publishing it.
    with (output/'self_test.log').open('w') as log:
        subprocess.run([str(binary),'--headless','--','--self-test'],stdout=log,
                       stderr=subprocess.STDOUT,timeout=45,check=True)
    # The build retains symbols; the redistributable can be stripped independently.
    subprocess.run(['strip','--strip-unneeded',str(output/required[2])],timeout=30,check=True)
    licenses=output/'licenses';licenses.mkdir()
    shutil.copyfile(project/font['license'].removeprefix('res://'),licenses/'Noto-CJK-LICENSE.txt')
    subprocess.run(['godot','--headless','--path',str(project),'--script',
        'res://standalone/license_dump.gd','--',str(licenses/'Godot.json')],
        stdout=subprocess.DEVNULL,timeout=30,check=True)
    ort=root/'native/.deps'/dependencies['onnxruntime']['directory']
    for name in ['LICENSE','ThirdPartyNotices.txt']:
        shutil.copyfile(ort/name,licenses/('ONNXRuntime-'+name))
    shutil.copyfile(root/'native/.deps/godot-cpp/LICENSE.md',licenses/'godot-cpp-LICENSE.md')
    for name in ['LICENSE','NOTICE']:
        shutil.copyfile(root/name,licenses/('MicroDuck-project-'+name))
    # all_resources also includes the optional visual assets. Keep their
    # supplied notices alongside the package, including per-asset credits.
    vendor=project/'vendor'
    if vendor.exists():
        for source in sorted(vendor.rglob('*')):
            if source.is_file() and source.name.lower() in ['license','license.md','license.txt','notice','copying']:
                target=licenses/'vendor'/source.relative_to(vendor)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    robot=load_robot_json(root/'robots/microduck_roller.json')
    upstream=next((parent for parent in Path(robot['mjcf']).parents if (parent/'pyproject.toml').is_file()),None)
    if upstream is None or not (upstream/'LICENSE').is_file():
        raise RuntimeError('Robot source license is required for the standalone package')
    shutil.copyfile(upstream/'LICENSE',licenses/'microduck_rl-LICENSE')
    (licenses/'robot-assets.txt').write_text('Microduck robot assets originate from https://github.com/pollen-robotics/microduck_rl\n'
        'The upstream README declares 3D model files under Creative Commons BY-SA-NC.\n'
        'Source software license: microduck_rl-LICENSE. Project and asset notices: MicroDuck-project-NOTICE.\n')
    atomic_json(output/'models.json',{skill:dict(sha256=item['sha256'],manifest=item['manifest'])
                                     for skill,item in deployment['policies'].items()})
    atomic_json(output/'build.json',dict(created_unix=time.time(),editor=version,dependencies=dependencies,
        git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        source_sha256=source_hashes(root,project),deployment=deployment,project=str(project),
        robot_resource_sha256={item[key]:sha256(project/item[key].removeprefix('res://'))
                              for item in deployment['robots'].values() for key in ['scene','spec']},
        self_test_sha256=sha256(project/'runtime_assets/self_test.json'),
        release_status='development_candidate; acceptance is recorded separately'))
    (output/'README.txt').write_text('''MicroDuck — Linux ARM64 standalone

Ubuntu 24.04 ARM64. Keep every file in this directory together.
Run: ./MicroDuck.arm64
No Python, source checkout, network or control server is required.

W/A/S/D: move and turn; Space: neutral; 6: walk/roller robot; 7: stand.
Y: sit/rise (roller: crouch); G: ground pickup; K/L: left/right kick;
R: forward roll; 0: reset; F8: pause/resume; Escape: exit.
The on-screen buttons show the actual shortcut mapping.

Embedded model numeric check: ./MicroDuck.arm64 --headless -- --self-test
Replay: ./MicroDuck.arm64 --headless --fixed-fps 200 -- --replay=/absolute/case.json --trace=/absolute/trace.json
Integrity: sha256sum -c SHA256SUMS

This is a development candidate until the separate acceptance report passes.
Model hashes and provenance are in models.json / build.json. Licenses are in licenses/.
''')
    files=sorted(p for p in output.rglob('*') if p.is_file())
    if 'sprint' in deployment['policies']:
        readme=output/'README.txt'
        readme.write_text(readme.read_text().replace('W/A/S/D: move and turn;',
            'Left Shift + W: walking sprint; A/D turn while sprinting. Right Shift does not activate sprint.\nW/A/S/D: move and turn;'))
    (output/'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.relative_to(output)}\n' for p in files))
    if archive:
        archive_path=output.with_suffix('.tar.gz')
        if archive_path.exists():raise FileExistsError(archive_path)
        with tarfile.open(archive_path,'w:gz') as target:target.add(output,arcname=output.name)
    return dict(directory=str(output),executable=str(binary),models={k:v['sha256'] for k,v in deployment['policies'].items()},
                size_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('--archive',action='store_true');p.add_argument('--project',type=Path);a=p.parse_args()
    print(json.dumps(package(a.output,a.archive,a.project)))
