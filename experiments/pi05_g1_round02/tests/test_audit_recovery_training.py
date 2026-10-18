"""Small synthetic post-run acceptance; no real observations/model payloads."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.pi05_g1_round02.audit_recovery_training import audit,optimizer_archive_receipt
from experiments.pi05_g1_round02.checkpoint_lineage import ROSTER_SCHEMA,sha
from experiments.pi05_g1_round02.tests import test_checkpoint_lineage as lineage_fixture
write=lineage_fixture.write


class RecoveryAcceptanceTests(unittest.TestCase):
    def fixture(self,root):
        import torch
        helper=lineage_fixture.CheckpointLineageTests();f=helper.fixture(root)
        for path in f.old.glob('*/metadata.json'):
            value=json.loads(path.read_text());value['model_dtype_counts']={'torch.float32':{'parameter_count':4,'parameter_bytes':16}};write(path,value)
        baseline=helper.recovery(f,0,.4)
        (baseline/'model.safetensors').unlink();os.link(f.parent/'model.safetensors',baseline/'model.safetensors')
        base=json.loads((baseline/'metadata.json').read_text());base.update(optimizer_payload_saved=False);write(baseline/'metadata.json',base)
        periodic=helper.recovery(f,100,.3);pm=json.loads((periodic/'metadata.json').read_text())
        cleanup={'phase':'after_model_serialize','before':{'VmRSS_bytes':100000,'RssAnon_bytes':90000,'MemAvailable_bytes':1000000},'after':{'VmRSS_bytes':70000,'RssAnon_bytes':60000,'MemAvailable_bytes':1030000},'trim_available':True,'trim_return':1}
        pm.update(optimizer_payload_saved=False,cpu_staging_cleanup=[cleanup]);write(periodic/'metadata.json',pm)
        final=f.new/'final_000100';final.mkdir();(final/'model.safetensors').write_bytes(b'next tiny weights')
        fm=copy.deepcopy(pm);fm.update(optimizer_payload_saved=True,cpu_staging_cleanup=[cleanup,{**cleanup,'phase':'after_optimizer_serialize'}]);write(final/'metadata.json',fm)
        torch.save({'optimizer':{'schema':'fp32_master_adamw_v1','master_parameters':[torch.ones(2)],'optimizer':{'exp_avg':torch.zeros(2),'exp_avg_sq':torch.ones(2)}},'step':100,'round02_data_identity':f.identity},final/'training_state.pt')
        write(f.new/'candidate_training_dirs.json',{'schema':ROSTER_SCHEMA,'training_dirs':[str(f.old),str(f.new)],'current_training_dir':str(f.new)})
        write(f.new/'best_checkpoint.json',{'path':str(periodic),'validation_flow_loss':.3});write(f.new/'last_checkpoint.json',{'path':'final_000100'})
        events=[{'event':'cpu_admission_complete','round02_data_identity':f.identity},{'event':'training_started','config':{'batch_size':1,'gradient_accumulation':8,'learning_rate':1e-5,'head_only':False,'checkpoint_every':100},'optimizer_master_dtype':'float32','warmstart':base['warmstart']}]
        counts={'original':{'prefix':0,'full':0},'fresh_teacher':{'prefix':0,'full':0}}
        for step in range(1,101):
            counts['original']['full']+=4;counts['fresh_teacher']['prefix']+=3;counts['fresh_teacher']['full']+=1
            events.append({'event':'optimizer_step','step':step,'flow_loss':.1,'gradient_norm':.2,'step_seconds':.01,'inherited_round_updates':200,'cumulative_round_updates':200+step,'parameter_update_probes':{name:{'changed_elements':1} for name in ('vision','VLM','expert','action_in_proj','action_out_proj')},'batch_sources':['original']*4+['fresh_teacher']*4,'time_sampling_draw_counts':{'original':{'prefix':0,'full':4},'fresh_teacher':{'prefix':3,'full':1}},'cumulative_time_sampling_draw_counts':copy.deepcopy(counts)})
        events.append({'event':'training_stopped','step':100,'cumulative_round_updates':300,'final_checkpoint':'final_000100','selected_checkpoint':str(periodic),'best_validation_loss':.3})
        (f.new/'progress.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in events));f.events=events;return f

    def run_audit(self,f):return audit(f.new,f.config,f.norm,f.loss)

    def test_complete_synthetic_acceptance_counts_lost_updates_and_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));report=self.run_audit(f)
            self.assertEqual(report['status'],'passed_cpu_acceptance',report['errors'])
            self.assertEqual(report['update_accounting']['failed_attempt_unpersisted_completed'],143)
            self.assertEqual(report['update_accounting']['retained_lineage_cumulative'],300)
            self.assertEqual(report['update_accounting']['all_known_executed_completed'],443)
            self.assertFalse(report['optimizer_archive']['tensor_payload_read']);self.assertEqual(len(report['trim_observations']),1)

    def test_numeric_baseline_delta_diagnostic_without_posthoc_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));path=f.new/'step_000000/metadata.json';m=json.loads(path.read_text())
            m['validation_flow_loss']-=.000025
            for metric in m['validation_sources'].values():metric['flow_loss']-=.000025
            write(path,m);report=self.run_audit(f)
            self.assertEqual(report['status'],'passed_cpu_acceptance',report['errors']);self.assertFalse(report['baseline']['losses_bitwise_equal'])
            self.assertLess(report['baseline']['combined_difference'],0)

    def test_no_true_stop_or_final_optimizer_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));(f.new/'progress.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in f.events[:-1]));report=self.run_audit(f)
            self.assertEqual(report['status'],'incomplete_or_failed');self.assertIn('exactly_one_true_training_stopped',report['errors'])
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));(f.new/'final_000100/training_state.pt').unlink();report=self.run_audit(f)
            self.assertEqual(report['status'],'incomplete_or_failed')

    def test_source_or_trim_accounting_and_invalid_archive_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));f.events[2]['cumulative_round_updates']=0
            (f.new/'progress.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in f.events));report=self.run_audit(f)
            self.assertIn('step_lineage_counter_consistency',report['errors'])
            path=f.new/'step_000100/metadata.json';m=json.loads(path.read_text());m['cpu_staging_cleanup']=[];write(path,m)
            self.assertIn('periodic_100_trim_diagnostics_present',self.run_audit(f)['errors'])
            bad=f.root/'bad.pt';bad.write_bytes(b'not a complete optimizer zip')
            with self.assertRaises(Exception):optimizer_archive_receipt(bad)


if __name__=='__main__':unittest.main()
