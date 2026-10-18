"""008 loaded diagnostic targets. No stepping or live state mutation here."""
from pathlib import Path
import importlib.util,json
import numpy as np
F=Path('/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_probe_006/full_body_readiness')
class ColdAnchoredLoadedTeacher:
    def __init__(self,cold_rod_world):
        spec=importlib.util.spec_from_file_location('sealed006_provider',F/'target_provider.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.provider=module.DoubleStanceBodyBackTeacher(static_bias=True,loaded=True)
        rows=json.loads(Path('/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_whole_body_007/l1_bias_proposal.json').read_text());by={r['name']:r for r in rows}
        for frame in self.provider.frames:frame['tau_static_0N18_Nm']=by[frame['phase']]['new_tau18']
        self.anchor=np.asarray(cold_rod_world,float).copy();self.reference_root0=np.asarray(self.provider.frames[0]['expected_root_xyz_offline_reference_only'],float)
    def action(self,env,t,actual_rod_world=None):
        nominal,tau,rootref,virtualtcp=self.provider.interpolate(t)
        fixed_target=self.anchor+np.array([rootref[0]-self.reference_root0[0],0.,0.])
        action,metadata=self.provider.action(env,t,fixed_target)
        metadata.update(cold_rod_anchor_world=self.anchor.tolist(),fixed_cold_anchor_based_TCP_target=fixed_target.tolist(),actual_rod_monitor_only=None if actual_rod_world is None else np.asarray(actual_rod_world).tolist(),does_not_reanchor_to_current_rod=True,nominal_target18=nominal.tolist(),static_tau_bias18=tau.tolist(),cold_prepositioned_real_contact_not_active_grasp=True)
        return action,metadata
