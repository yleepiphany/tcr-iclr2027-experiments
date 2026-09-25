"""No GPU subprocesses: fake snapshots, real temporary flock files and race tests."""
import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import safe_successor_v2 as m

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.queue=self.root/'queue';self.queue.mkdir()
        self.empty={'gpu':2,'uuid':'GPU-test','free_mib':81153,'compute_pids':[]}
    def tearDown(self):self.tmp.cleanup()
    def disks(self):return patch.object(m.shutil,'disk_usage',return_value=type('Disk',(),{'free':200*1024**3})())
    def test_occupied_training_or_evaluator_never_enters(self):
        for pids in (['100'],['101','102']):
            with patch.object(m,'snapshot',return_value={**self.empty,'compute_pids':pids}):
                with m.acquire_empty_card(self.root,2,self.queue) as result:self.assertIsNone(result)
        self.assertFalse((self.root/'vla-merge-runtime').exists())
    def test_memory_floor_and_new_process_race_reject(self):
        for second in ({**self.empty,'free_mib':49151},{**self.empty,'compute_pids':['new-job']},{**self.empty,'uuid':'GPU-swapped'}):
            with patch.object(m,'snapshot',side_effect=[self.empty,second]),self.disks():
                with m.acquire_empty_card(self.root,2,self.queue) as result:self.assertIsNone(result)
    def test_real_lease_blocks_and_success_releases_both(self):
        path=m.lock_paths(self.root,2,self.empty['uuid'])[0];path.parent.mkdir(parents=True)
        with path.open('a') as owner:
            fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch.object(m,'snapshot',return_value=self.empty),self.disks():
                with m.acquire_empty_card(self.root,2,self.queue) as result:self.assertIsNone(result)
        with patch.object(m,'snapshot',return_value=self.empty),self.disks():
            with m.acquire_empty_card(self.root,2,self.queue) as result:
                self.assertEqual(len(result['fds']),2)
                for lock in m.lock_paths(self.root,2,self.empty['uuid']):
                    with lock.open('a') as challenger:
                        with self.assertRaises(BlockingIOError):fcntl.flock(challenger,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for lock in m.lock_paths(self.root,2,self.empty['uuid']):
                with lock.open('a') as next_owner:fcntl.flock(next_owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def test_child_keeps_inherited_leases_after_parent_closes(self):
        import subprocess,sys
        child=None
        try:
            with patch.object(m,'snapshot',return_value=self.empty),self.disks():
                with m.acquire_empty_card(self.root,2,self.queue) as held:
                    script="import fcntl,sys; [fcntl.flock(int(fd),fcntl.LOCK_EX|fcntl.LOCK_NB) for fd in sys.argv[1:]]; print('ready',flush=True); sys.stdin.read()"
                    child=subprocess.Popen([sys.executable,'-c',script,*map(str,held['fds'])],pass_fds=held['fds'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
                    self.assertEqual(child.stdout.readline().strip(),'ready')
                for lock in m.lock_paths(self.root,2,self.empty['uuid']):
                    with lock.open('a') as challenger:
                        with self.assertRaises(BlockingIOError):fcntl.flock(challenger,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:
            if child is not None:
                child.stdin.close();self.assertEqual(child.wait(timeout=10),0);child.stdout.close()
        for lock in m.lock_paths(self.root,2,self.empty['uuid']):
            with lock.open('a') as next_owner:fcntl.flock(next_owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def test_disk_shortage_does_not_admit(self):
        with patch.object(m,'snapshot',return_value=self.empty),patch.object(m.shutil,'disk_usage',return_value=type('Disk',(),{'free':1})()):
            with self.assertRaises(RuntimeError):
                with m.acquire_empty_card(self.root,2,self.queue):pass
    def test_stage_and_successor_duplicate_owner_rejected(self):
        for stage,group in [('smoke','all'),('teachers','coordination'),('solve','all'),('supervisor','successor')]:
            with m.stage_guard(self.queue,stage,group):
                with self.assertRaises(BlockingIOError):
                    with m.stage_guard(self.queue,stage,group):pass
    def test_source_contains_no_process_signal_calls(self):
        import ast
        tree=ast.parse(Path(m.__file__).read_text())
        dangerous={'kill','killpg','terminate','send_signal'}
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in dangerous for n in ast.walk(tree)))
if __name__=='__main__':unittest.main()
