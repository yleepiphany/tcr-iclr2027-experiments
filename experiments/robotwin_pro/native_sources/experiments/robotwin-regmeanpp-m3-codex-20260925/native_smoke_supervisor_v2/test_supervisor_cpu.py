import fcntl,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import supervisor as s

class Lock:
    def __init__(self,path):
        self.path=Path(path);self.handle=os.open(path,os.O_CREAT|os.O_RDWR,0o600);self._released=False
        fcntl.flock(self.handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def release(self):raise AssertionError('LOCK_UN release must not be used for inherited leases')

def available(path):
    fd=os.open(path,os.O_RDWR)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        fcntl.flock(fd,fcntl.LOCK_UN);return True
    finally:os.close(fd)

class Clock:
    def __init__(self):self.t=0
    def time(self):return self.t
    def sleep(self,n):self.t+=n

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.held=[]
        self.idle={'gpu':4,'uuid':s.GPUS[4],'free_mib':81153,'used_mib':1,'utilization':0,'compute_pids':[]}
    def tearDown(self):
        s.close_own(self.held);self.temp.cleanup()
    def locks(self):
        result=[Lock(self.root/f'lease-{len(self.held)+i}.lock') for i in range(3)];self.held.extend(result);return result
    def test_valid_three_descriptors_verified(self):
        locks=self.locks();s.verify_inherited_leases(s.lease_records(locks),[str(x.path) for x in locks])
        self.assertTrue(all(not available(x.path) for x in locks))
    def test_descriptor_inode_tampering_rejected(self):
        locks=self.locks();records=s.lease_records(locks);records[0]['inode']+=1
        with self.assertRaises(ValueError):s.verify_inherited_leases(records,[str(x.path) for x in locks])
    def test_foreign_open_description_rejected_even_same_inode(self):
        locks=self.locks();records=s.lease_records(locks);foreign=os.open(locks[0].path,os.O_RDWR)
        records[0]['fd']=foreign
        try:
            with self.assertRaises(ValueError):s.verify_inherited_leases(records,[str(x.path) for x in locks])
        finally:os.close(foreign)
    def test_lost_lease_rejected(self):
        locks=self.locks();records=s.lease_records(locks);fcntl.flock(locks[0].handle,fcntl.LOCK_UN)
        with self.assertRaises(ValueError):s.verify_inherited_leases(records,[str(x.path) for x in locks])
    def test_unexpected_resource_paths_rejected(self):
        locks=self.locks();records=s.lease_records(locks)
        with self.assertRaises(ValueError):s.verify_inherited_leases(records,['wrong',str(locks[1].path),str(locks[2].path)])
    def test_inherited_fds_prevent_competitor_after_parent_closes(self):
        locks=self.locks();records=s.lease_records(locks);paths=[str(x.path) for x in locks]
        child_code='''import json,sys
sys.path.insert(0,sys.argv[1]);import supervisor as s
r=json.loads(sys.argv[2]);s.verify_inherited_leases(r,[x['path'] for x in r])
print('HELD',flush=True);sys.stdin.readline()
s.verify_inherited_leases(r,[x['path'] for x in r]);print('STILL_HELD',flush=True)
'''
        child=subprocess.Popen([sys.executable,'-c',child_code,str(Path(s.__file__).parent),json.dumps(records)],
            pass_fds=tuple(x.handle for x in locks),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(),'HELD')
            self.assertTrue(all(not available(p) for p in paths))
            s.close_own(locks)  # Simulates supervisor exit without unlocking the inherited OFDs.
            self.assertTrue(all(not available(p) for p in paths))
            child.stdin.write('finish\n');child.stdin.flush()
            self.assertEqual(child.stdout.readline().strip(),'STILL_HELD')
            self.assertEqual(child.wait(timeout=10),0)
            self.assertTrue(all(available(p) for p in paths))
        finally:
            if child.poll() is None:
                child.stdin.write('finish\n');child.stdin.flush();child.wait(timeout=10)
            child.stdin.close();child.stdout.close();child.stderr.close()
    def test_acquire_retains_locks_through_return(self):
        locks=self.locks();clock=Clock();rows=iter([self.idle,self.idle])
        held,receipt=s.wait_and_hold(4,s.GPUS[4],board_fn=lambda g:next(rows),admissible_fn=lambda row,g:True,
            acquire_fn=lambda g,u:locks,check_fn=lambda:None,sleep_fn=clock.sleep,clock=clock.time)
        self.assertEqual(held,locks);self.assertTrue(receipt['leases_retained'])
        self.assertTrue(all(not available(x.path) for x in held))
    def test_race_second_check_closes_own_then_waits(self):
        first=self.locks();second=self.locks();clock=Clock();busy={**self.idle,'compute_pids':[42]}
        rows=iter([self.idle,busy,self.idle,self.idle]);sets=iter([first,second])
        held,_=s.wait_and_hold(4,s.GPUS[4],board_fn=lambda g:next(rows),admissible_fn=lambda r,g:not r['compute_pids'],
            acquire_fn=lambda g,u:next(sets),check_fn=lambda:None,sleep_fn=clock.sleep,clock=clock.time)
        self.assertEqual(clock.t,15);self.assertEqual(held,second)
        self.assertTrue(all(available(x.path) for x in first));self.assertTrue(all(not available(x.path) for x in second))
    def test_busy_and_timeout_never_acquire(self):
        clock=Clock();busy={**self.idle,'compute_pids':[42]};calls=[]
        with self.assertRaises(TimeoutError):
            s.wait_and_hold(4,s.GPUS[4],board_fn=lambda g:busy,admissible_fn=lambda r,g:False,
                acquire_fn=lambda g,u:calls.append(True),check_fn=lambda:None,sleep_fn=clock.sleep,clock=clock.time,timeout=30)
        self.assertFalse(calls)
    def test_partial_lock_acquisition_releases_owned_fds(self):
        for missing in (0,1,2):
            made=[]
            class Flocks:
                def take(_self,*args,**kwargs):
                    if len(made)==missing:return None
                    lock=Lock(self.root/f'partial-{missing}-{len(made)}.lock');made.append(lock);return lock
                take_card=take;_try_lock=take;take_build_slot=take
            self.assertIsNone(s.acquire_all(Flocks(),4,s.GPUS[4]));self.assertTrue(all(available(x.path) for x in made))
    def test_failure_after_full_lockset_releases_owned_fds(self):
        locks=self.locks();clock=Clock();rows=iter([self.idle,{**self.idle,'uuid':'wrong'}])
        with self.assertRaises(ValueError):
            s.wait_and_hold(4,s.GPUS[4],board_fn=lambda g:next(rows),admissible_fn=lambda r,g:True,
                acquire_fn=lambda g,u:locks,check_fn=lambda:None,sleep_fn=clock.sleep,clock=clock.time)
        self.assertTrue(all(available(x.path) for x in locks))
    def test_child_guard_has_only_original_smoke_call(self):
        import ast
        tree=ast.parse((Path(s.__file__).parent/'child_guard.py').read_text())
        imports=[n for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module=='materialize_regmeanpp_m3']
        self.assertEqual([[a.name for a in n.names] for n in imports],[['smoke']])
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('kill','terminate','send_signal') for n in ast.walk(tree)))
    def test_core_claim_and_each_attempt_marker_reject_duplicate(self):
        import child_guard
        for index,name in enumerate(('smoke-PERMIT-CONSUMED.json','native-smoke.json','smoke-FAILED.json')):
            core=self.root/f'claimed-{index}';core.mkdir();(core/name).write_text('{}')
            with self.assertRaises(FileExistsError):child_guard.claim_once(core,{'plan_sha256':'frozen'})
        core=self.root/'fresh';core.mkdir()
        child_guard.claim_once(core,{'plan_sha256':'frozen','pid':123})
        with self.assertRaises(FileExistsError):child_guard.claim_once(core,{'plan_sha256':'frozen','pid':999})
        self.assertEqual(json.loads((core/'smoke-PERMIT-CONSUMED.json').read_text())['pid'],123)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps({'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'gpu_used':False,'gpu_queries':False,
        'cpu_lock_inheritance_subprocesses':1,'native_smoke_children_launched':0,'signals_sent':0}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
