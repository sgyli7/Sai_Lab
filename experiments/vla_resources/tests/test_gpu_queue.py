import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from experiments.vla_resources import gpu_queue as q

DUMMY = '''import argparse,json,os,signal,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--ledger');p.add_argument('--stage');p.add_argument('--max-seconds');p.add_argument('--delay',type=float,default=.15);p.add_argument('--code',type=int,default=0);p.add_argument('--unknown',action='store_true');p.add_argument('--events');a=p.parse_args()
path=Path(a.ledger);doc={'runs':[{'status':'running'}]};path.write_text(json.dumps(doc));stopped=False
def stop(*args):
 global stopped
 stopped=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def event(name):
 with open(a.events,'a') as f:f.write(json.dumps({'event':name,'ledger':str(path),'time':time.monotonic()})+'\\n')
event('start');end=time.monotonic()+a.delay
while time.monotonic()<end and not stopped:time.sleep(.01)
row=doc['runs'][-1];row.update(status='failed' if a.code or stopped else 'completed',last_unit_properties={'ActiveState':'inactive'},docker_cleanup={'complete':not a.unknown},accounting_open=a.unknown)
path.write_text(json.dumps(doc));event('end');raise SystemExit(143 if stopped else a.code)
'''

class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'lingbot_request.json').write_text(json.dumps({'thread_id':'lingbot','peer_thread_id':'pi'}))
        (self.root/'legacy_barrier.json').write_text('{"active":false}')
        self.dummy=self.root/'dummy.py';self.dummy.write_text(DUMMY)
        self.events=self.root/'events.jsonl';self.processes=[]
    def tearDown(self):
        for p in self.processes:
            if p.poll() is None:p.terminate()
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            if p.stdout:p.stdout.close()
            if p.stderr:p.stderr.close()
        self.temp.cleanup()
    def ticket(self,owner='lingbot'):
        return q.request(self.root,owner,'preflight',10)['ticket']
    def command(self,ledger,extra=()):
        return [sys.executable,str(self.dummy),'--ledger',str(ledger),'--stage','preflight','--max-seconds','10','--events',str(self.events),*extra]
    def start(self,ticket,extra=()):
        ledger=self.root/(ticket+'.json')
        argv=[sys.executable,'-m','experiments.vla_resources.gpu_queue','--coord-root',str(self.root),'run','--ticket',ticket,'--ledger',str(ledger),'--wait-seconds','3','--',*self.command(ledger,extra)]
        p=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);self.processes.append(p);return p
    def wait_running(self,ticket):
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            rows=q.status(self.root)['tickets']
            if next(r for r in rows if r['ticket']==ticket)['status']=='running' and (self.root/(ticket+'.json')).exists():return
            time.sleep(.01)
        self.fail('producer did not start')
    def test_fifo_and_exclusion(self):
        a=self.ticket();b=self.ticket('pi');pb=self.start(b);time.sleep(.12)
        self.assertFalse((self.root/(b+'.json')).exists())
        pa=self.start(a);self.assertEqual(pa.wait(timeout=4),0);self.assertEqual(pb.wait(timeout=4),0)
        events=[json.loads(s) for s in self.events.read_text().splitlines()]
        self.assertEqual([x['event'] for x in events],['start','end','start','end'])
        self.assertIn(a,events[0]['ledger']);self.assertIn(b,events[2]['ledger'])
    def test_barrier_keeps_wait_and_ledger_untouched(self):
        a=self.ticket();(self.root/'legacy_barrier.json').write_text('{"active":true}')
        ledger=self.root/'ledger.json';ledger.write_text('{"runs":[]}');original=ledger.read_bytes()
        self.assertEqual(q.run(self.root,a,ledger,self.command(ledger),.08),75)
        self.assertEqual(ledger.read_bytes(),original)
        self.assertEqual(q.status(self.root)['tickets'][0]['status'],'waiting')
    def test_failure_with_cleanup_releases_next(self):
        a=self.ticket();b=self.ticket('pi');pa=self.start(a,('--code','3'));pb=self.start(b)
        self.assertEqual(pa.wait(timeout=4),3);self.assertEqual(pb.wait(timeout=4),0)
        self.assertEqual([r['status'] for r in q.status(self.root)['tickets']],['failed','completed'])
    def test_unknown_cleanup_blocks_next(self):
        a=self.ticket();b=self.ticket('pi');pa=self.start(a,('--unknown',));self.assertEqual(pa.wait(timeout=4),70)
        ledger=self.root/(b+'.json');self.assertEqual(q.run(self.root,b,ledger,self.command(ledger),.08),75)
        self.assertFalse(ledger.exists());self.assertEqual(q.status(self.root)['tickets'][0]['status'],'cleanup_unknown')
    def test_cancel_owner(self):
        a=self.ticket()
        with self.assertRaises(ValueError):q.cancel(self.root,a,'pi')
        q.cancel(self.root,a,'lingbot');self.assertEqual(q.status(self.root)['tickets'][0]['status'],'cancelled')
    def test_duplicate_run_rejected(self):
        a=self.ticket();pa=self.start(a,('--delay','.5'));self.wait_running(a);pb=self.start(a)
        self.assertNotEqual(pb.wait(timeout=3),0);self.assertEqual(pa.wait(timeout=3),0)
    def test_signal_waits_for_supervisor_cleanup(self):
        a=self.ticket();pa=self.start(a,('--delay','2'));self.wait_running(a);pa.send_signal(signal.SIGTERM)
        self.assertEqual(pa.wait(timeout=3),143)
        row=q.status(self.root)['tickets'][0];self.assertEqual(row['status'],'failed');self.assertTrue(row['cleanup_confirmed'])
    def test_duration_and_ledger_contract_checked_before_launch(self):
        a=self.ticket();ledger=self.root/'one.json';cmd=self.command(ledger);cmd[cmd.index('--max-seconds')+1]='11'
        with self.assertRaises(ValueError):q.run(self.root,a,ledger,cmd,0)
        self.assertFalse(ledger.exists());self.assertEqual(q.status(self.root)['tickets'][0]['status'],'waiting')

if __name__=='__main__':unittest.main()
