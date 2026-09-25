import json,tempfile,unittest
from pathlib import Path
import supervisor as s

class Clock:
    def __init__(self):self.t=0
    def time(self):return self.t
    def sleep(self,t):self.t+=t
class Tests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.uuid=s.GPUS[1]
        self.idle={'gpu':1,'uuid':self.uuid,'free_mib':81153,'used_mib':1,'utilization':0,'compute_pids':[]}
    def wait(self,boards,leases=lambda g,u:True,check=lambda:None,timeout=30):
        iterator=iter(boards)
        return s.wait_for_admission(1,self.uuid,board_fn=lambda g:next(iterator),
            admissible_fn=lambda r,g:r['free_mib']>=71680 and r['used_mib']<=64 and r['utilization']==0 and not r['compute_pids'],
            leases_fn=leases,check_fn=check,sleep_fn=self.clock.sleep,clock=self.clock.time,poll_seconds=15,maximum_wait_seconds=timeout)
    def test_empty_twice_and_available_leases(self):
        result=self.wait([self.idle,self.idle]);self.assertEqual(result['board_checks'],2)
    def test_busy_card_waits_no_launch(self):
        busy={**self.idle,'compute_pids':[1],'free_mib':40000}
        result=self.wait([busy,self.idle,self.idle]);self.assertEqual(result['wait_seconds'],15)
    def test_lease_contended_waits(self):
        calls=[]
        def leases(g,u):calls.append(1);return len(calls)>1
        result=self.wait([self.idle,self.idle,self.idle],leases)
        self.assertEqual(result['wait_seconds'],15)
    def test_second_card_check_loses_race_then_waits(self):
        busy={**self.idle,'compute_pids':[42]}
        result=self.wait([self.idle,busy,self.idle,self.idle]);self.assertEqual(result['wait_seconds'],15)
    def test_uuid_change_fails(self):
        with self.assertRaises(ValueError):self.wait([{**self.idle,'uuid':'wrong'}])
    def test_timeout_fails_without_child(self):
        busy={**self.idle,'used_mib':4096}
        with self.assertRaises(TimeoutError):self.wait([busy,busy,busy])
    def test_permit_or_claim_changed_fails(self):
        def changed():raise ValueError('Claim changed')
        with self.assertRaises(ValueError):self.wait([self.idle],check=changed)
    def test_command_is_only_smoke(self):
        p={'python':'python','runner':'runner.py','core_run':'/tmp/run'}
        cmd=s.command(p,1,'/tmp/permit.json')
        self.assertEqual(cmd[cmd.index('--stage')+1],'smoke');self.assertNotIn('materialize',cmd)
        with self.assertRaises(ValueError):s.command(p,0,'/tmp/permit.json')
    def test_probe_always_releases_all_acquired_handles(self):
        for missing in (0,1,2,3):
            acquired=[];released=[]
            class Lock:
                def __init__(self,i):self.i=i
                def release(self):released.append(self.i)
            class Fake:
                def take(self,*a,**kw):
                    i=len(acquired)
                    if i==missing:return None
                    acquired.append(i);return Lock(i)
                take_card=take;_try_lock=take;take_build_slot=take
            passed=s.probe_leases(Fake(),1,self.uuid)
            self.assertEqual(passed,missing==3);self.assertEqual(released,list(reversed(acquired)))
    def test_result_requires_all_three_groups_and_all_inputs(self):
        rows=[{'group':g,'replica':0,'physical_timestep':1.,'velocity_shape':[1,50,32],
            'sampled_inputs_bitwise':418,'native_velocity_bitwise':True,'candidate_blocks_bitwise':['vision0','language0','action0']}
            for g in ('coordination','receptacle','precision')]
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'native-smoke.json'
            result={'status':'PASS','plan_sha256':'abc','formal_episodes':0,'reports':rows}
            file.write_text(json.dumps(result));self.assertEqual(s.check_result(tmp,'abc')['status'],'PASS')
            rows[0]['sampled_inputs_bitwise']=417;file.write_text(json.dumps(result))
            with self.assertRaises(ValueError):s.check_result(tmp,'abc')

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps({'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'gpu_used':False,'gpu_queries':False,
        'children_launched':0,'signals_sent':0}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
