from pathlib import Path
import json,sys,unittest
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_stop_state_20260912')
loader=unittest.TestLoader();suite=unittest.TestSuite()
for pattern in json.loads((r/'test_protocol.json').read_text())['patterns']:
 suite.addTests(loader.discover('tests',pattern=pattern))
result=unittest.TextTestRunner(verbosity=1).run(suite)
atomic_json(r/'closeout/tests.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),successful=result.wasSuccessful()))
sys.exit(0 if result.wasSuccessful() else 1)
