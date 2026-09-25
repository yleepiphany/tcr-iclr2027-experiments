"""CPU-only queue protocol, resource, lease and per-episode audit tests."""
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import queue_featcal as q

class QueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master=q.read(q.FIG/'plan.json')
        cls.jobs=q.make_jobs(cls.master)
        cls.p={'host':q.HOST,'gpu_uuids':{str(g):f'GPU-test-{g}' for g in q.GPUS},'jobs':cls.jobs}

    def test_fixed_graph_and_denominator(self):
        self.assertEqual(len(self.jobs),9)
        self.assertEqual({s:sum(j.get('episodes',0) for j in self.jobs if j['subset']==s) for s in q.SUBSETS},
                         {'spatial-goal':100,'spatial-object-goal':300})
        self.assertEqual(sum(j['stage']=='teachers' for j in self.jobs),3)
        for s,groups in q.SUBSETS.items():
            if s=='spatial-object-goal':
                solve=next(j for j in self.jobs if j['id']==s+'-solve')
                self.assertEqual(solve['needs'],[f'{s}-teacher-{g}' for g in groups])
            for j in self.jobs:
                if j['subset']==s and j['stage']=='formal':self.assertEqual(j['needs'],[s+'-solve'])

    def permit(self,digest):
        return {'schema':'fig4_featcal_queue_execution_permit_v3','allowed':True,'queue_plan_sha256':digest,
            'master_plan_sha256':q.MASTER_SHA,'host':q.HOST,'gpu_uuids':self.p['gpu_uuids'],
            'job_ids':[j['id'] for j in self.jobs],'formal_episodes':400,'reused_formal_episodes':100}
    def test_permit_requires_exact_scope(self):
        good=self.permit('a'*64);q.validate_permit(good,self.p,'a'*64)
        for key,value in [('allowed',False),('host','other'),('formal_episodes',500),('job_ids',good['job_ids'][:-1]),('queue_plan_sha256','b'*64)]:
            bad=copy.deepcopy(good);bad[key]=value
            with self.assertRaises(ValueError):q.validate_permit(bad,self.p,'a'*64)
    def test_permit_consumption_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);q.write(root/'plan.json',self.p);permit=root/'permit.json';q.write(permit,self.permit(q.sha(root/'plan.json')))
            q.consume_permit(permit,self.p,root)
            with self.assertRaises(FileExistsError):q.consume_permit(permit,self.p,root)

    def test_original_build_backend_and_separate_formal_backend(self):
        with patch.dict(os.environ,{'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE':'inherited-wrong'}):
            for stage in ['smoke','teachers','solve']:
                self.assertEqual(q.environment(1,stage)['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'],'1')
            self.assertNotIn('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',q.environment(1,'formal'))
        with self.assertRaises(ValueError):q.environment(1,'unknown')

    def test_exact_resource_boundaries(self):
        row={'uuid':'GPU-test','free_mib':49152,'used_mib':0,'utilization':0,'compute':[]}
        self.assertTrue(q.resource_ok(row,120*2**30))
        for key,value in [('free_mib',49151),('used_mib',65),('utilization',1),('compute',['unknown-host-pid'])]:
            bad={**row,key:value};self.assertFalse(q.resource_ok(bad,120*2**30))
        self.assertFalse(q.resource_ok(row,120*2**30-1))

    def test_busy_or_unknown_card_never_invalidates_completed_artifact(self):
        busy={'uuid':'GPU-test','free_mib':50000,'used_mib':20000,'compute':['host-namespace-unknown'],'utilization':20}
        clear={'uuid':'GPU-test','free_mib':81153,'used_mib':1,'compute':[],'utilization':0}
        self.assertEqual(q.observe_card_after_exit(1,'GPU-test',snapshot=lambda:{1:busy})['status'],'busy_or_identity_changed')
        self.assertEqual(q.observe_card_after_exit(1,'GPU-test',snapshot=lambda:{1:clear})['status'],'idle')
        self.assertEqual(q.observe_card_after_exit(1,'wrong-uuid',snapshot=lambda:{1:clear})['status'],'busy_or_identity_changed')
        def broken():raise RuntimeError('driver query unavailable')
        self.assertEqual(q.observe_card_after_exit(1,'GPU-test',snapshot=broken)['status'],'query_unavailable')

    def test_completed_stage_accepted_before_busy_card_and_other_worker_untouched(self):
        import io
        from unittest.mock import Mock
        job=next(j for j in self.jobs if j['stage']=='teachers');jid=job['id']
        process=Mock(pid=333);process.poll.return_value=0;process.wait.return_value=0
        other=Mock(pid=444);other.poll.return_value=None
        events=[]
        class Lease:
            def release(self):events.append('released')
        with tempfile.TemporaryDirectory() as temporary,patch.object(q,'RUN',Path(temporary)):
            root=Path(temporary);result={'status':'complete','group':job['group'],'plan_sha256':q.SUBSET_SHAS[job['subset']]}
            q.write(root/'worker-results'/f'{jid}.json',{'job_id':jid,'subset_plan_sha256':q.SUBSET_SHAS[job['subset']],'result':result})
            current={'process':process,'job':job,'log':io.StringIO(),'leases':[Lease(),Lease()],'gpu':1,'uuid':'GPU-test'}
            active={jid:current,'other':{'process':other}};accepted={}
            def observe(*args):
                self.assertTrue((root/'accepted'/f'{jid}.json').is_file());self.assertIn(jid,accepted);events.append('observed')
                return {'status':'busy_or_identity_changed','new_admission_required':True,'signalled_processes':[]}
            with patch.object(q,'observe_card_after_exit',side_effect=observe),patch.object(q,'terminate_own') as terminate:
                self.assertTrue(q.finish_completed(jid,current,self.p,{},accepted,active));terminate.assert_not_called()
            self.assertEqual(events,['observed','released','released']);self.assertEqual(set(active),{'other'})
            other.terminate.assert_not_called();other.kill.assert_not_called()
            exit_receipt=q.read(root/'exits'/f'{jid}.json');self.assertTrue(exit_receipt['artifact_accepted'])

    def test_wrong_worker_identity_not_accepted_or_hidden_by_clear_card(self):
        import io
        from unittest.mock import Mock
        job=next(j for j in self.jobs if j['stage']=='teachers');jid=job['id'];process=Mock(pid=3);process.poll.return_value=0
        with tempfile.TemporaryDirectory() as temporary,patch.object(q,'RUN',Path(temporary)):
            root=Path(temporary);q.write(root/'worker-results'/f'{jid}.json',{'job_id':jid,'subset_plan_sha256':'wrong','result':{}})
            x={'process':process,'job':job,'log':io.StringIO(),'leases':[],'gpu':1,'uuid':'GPU-test'};accepted={};active={jid:x}
            with patch.object(q,'observe_card_after_exit') as observe:
                with self.assertRaises(ValueError):q.finish_completed(jid,x,self.p,{},accepted,active)
                observe.assert_not_called()
            self.assertEqual(accepted,{});self.assertFalse((root/'accepted').exists())

    def test_failure_writes_terminal_state_without_stale_active(self):
        from unittest.mock import Mock
        process=Mock(pid=333);x={'process':process,'job':{'stage':'smoke'},'gpu':1}
        with tempfile.TemporaryDirectory() as temporary,patch.object(q,'RUN',Path(temporary)),patch.object(q,'terminate_own') as cleanup:
            active={'test-stage':x};q.terminal_failure(ValueError('synthetic'),{},[],active)
            self.assertEqual(active,{});self.assertEqual(q.read(Path(temporary)/'state.json')['active'],{})
            self.assertEqual(q.read(Path(temporary)/'FAILED.json')['interrupted'][0]['pid'],333);cleanup.assert_called_once()

    def test_changed_historical_failure_or_model_binding_cannot_be_reused(self):
        with patch.object(q,'sha',return_value='not-the-recorded-sha'):
            with self.assertRaises(ValueError):q.recover_inputs(self.master)

    def test_graph_selection_is_independent_of_success_numbers(self):
        changed=copy.deepcopy(self.master);changed['ignored_observed_successes']=[0,100,88]
        self.assertEqual(q.make_jobs(changed),self.jobs)
        self.assertEqual([j['id'] for j in self.jobs if j['subset']=='spatial-goal'],['featcal-spatial-goal-libero_goal'])
        self.assertFalse(q.INHERITED_IDS.intersection(j['id'] for j in self.jobs))

    def test_parent_lock_can_be_inherited_without_recursive_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lease';handle=os.open(path,os.O_RDWR|os.O_CREAT,0o600)
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                second=os.open(path,os.O_RDWR)
                try:
                    with self.assertRaises(BlockingIOError):fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB)
                finally:os.close(second)
                command=[sys.executable,'-c',f'import fcntl,os; fcntl.flock({handle},fcntl.LOCK_EX|fcntl.LOCK_NB); assert os.getppid()=={os.getpid()}']
                self.assertEqual(subprocess.run(command,pass_fds=(handle,),env={**os.environ,'CUDA_VISIBLE_DEVICES':''}).returncode,0)
            finally:os.close(handle)

    def test_cleanup_stops_only_owned_process_before_unlock(self):
        events=[]
        with tempfile.TemporaryDirectory() as tmp:
            log=(Path(tmp)/'owned.log').open('w')
            owned=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],stdout=log,stderr=log)
            other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
            class Lease:
                def release(self):
                    if owned.poll() is None:raise AssertionError('Unlocked while child alive')
                    events.append('released')
            try:
                q.terminate_own([{'process':owned,'log':log,'leases':[Lease(),Lease()]}])
                self.assertEqual(owned.returncode,-signal.SIGTERM)
                self.assertIsNone(other.poll())
                self.assertEqual(events,['released','released'])
            finally:
                if owned.poll() is None:owned.kill();owned.wait()
                other.terminate();other.wait()

    def test_stop_or_pending_signal_blocks_launch(self):
        for stopped,pending in [(True,set()),(False,{signal.SIGTERM})]:
            with patch.object(q,'STOP',stopped),patch.object(q.signal,'pthread_sigmask',return_value=set()), \
                 patch.object(q.signal,'sigpending',return_value=pending),patch.object(q.subprocess,'Popen') as popen:
                with self.assertRaises(ValueError):q.spawn_owned(['never-run'],{},None,[])
                popen.assert_not_called()

    def test_real_reset_audit_accepts_100_and_rejects_corruption(self):
        selection=Path(self.master['evaluation_selection']['path']);bank=selection.parent.parent
        selected=q.read(selection)['tasks'];manifest=q.read(bank/'manifest.json')['tasks']
        p={'selection':{'selection':str(selection),'bank':str(bank),'sha256':q.sha(selection),'bank_manifest_sha256':q.sha(bank/'manifest.json')},
           'assets':{str(q.REPO/'scripts/eval_pi05_policy_with_procedural_bank.py'):q.sha(q.REPO/'scripts/eval_pi05_policy_with_procedural_bank.py')}}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'eval';out.mkdir();job={'id':'cpu-only-fixture','subset':'spatial-goal','suite':'libero_spatial','output':tmp}
            info={'per_task':[{'task_id':i,'task_group':'libero_spatial','metrics':{'successes':[i*10+j<37 for j in range(10)]}} for i in range(10)],
                  'overall':{'n_episodes':100,'pc_success':37.}}
            reset={'eval_seed':274001,'repeat_id':'repeat-01','selection_sha256':p['selection']['sha256'],'bank_manifest_sha256':p['selection']['bank_manifest_sha256'],
                   'entrypoint_sha256':next(iter(p['assets'].values())),'tasks':{}}
            for i in range(10):
                k=f'libero_spatial/{i:02d}';indices=selected[k]
                reset['tasks'][k]={'suite':'libero_spatial','task_id':i,'task_name':manifest[k]['task_name'],'state_indices':indices,
                    'raw_state_sha256':[manifest[k]['states'][n]['raw_sha256'] for n in indices]}
            def publish():
                (out/'eval_info.json').write_text(json.dumps(info));reset['eval_info_sha256']=q.sha(out/'eval_info.json')
                (out/'procedural_bank_receipt.json').write_text(json.dumps(reset))
            publish();q.write(out/'bounded_resources.json',{'evaluation_completed':True,'changes_policy_rng':False})
            result=q.audit_formal(job,p,{'sha256':'synthetic-not-a-model'})
            self.assertEqual((result['successes'],result['episodes']),(37,100))
            info['per_task'][0]['metrics']['successes'].pop();publish()
            with self.assertRaises(ValueError):q.audit_formal(job,p,{'sha256':'synthetic-not-a-model'})
            info['per_task'][0]['metrics']['successes'].append(True)
            reset['tasks']['libero_spatial/00']['task_name']='wrong';publish()
            with self.assertRaises(ValueError):q.audit_formal(job,p,{'sha256':'synthetic-not-a-model'})

if __name__=='__main__':unittest.main()
