import copy
import json
from pathlib import Path
import tempfile
import unittest
import queue_regmeanpp_subset_v3 as queue

class QueueTests(unittest.TestCase):
    def setUp(self):self.plan=queue.read(queue.RUN/'plan.json')
    def test_exact_parent_jobs_and_no_four_expert_rebuild(self):
        queue.validate_plan(self.plan,False)
        figure=queue.frozen_figure()
        original={x['id']:x for x in figure['new_evaluations'] if x['method']=='regmeanpp'}
        self.assertEqual(len(self.plan['builds']),2);self.assertEqual(len(self.plan['evaluations']),5)
        self.assertEqual(sum(j['episodes'] for j in self.plan['evaluations']),500)
        for job in self.plan['evaluations']:self.assertEqual(job['command_template'],original[job['id']]['command_template'])
        self.assertTrue(self.plan['four_expert_endpoint_reused_not_queued'])
    def test_disallowed_or_wrong_scope_permit_rejected(self):
        template=queue.read(queue.RUN/'PERMIT-TEMPLATE-NOT-AUTHORIZED.json');sha=queue.digest(queue.RUN/'plan.json')
        with self.assertRaises(ValueError):queue.validate_permit(template,sha)
        good={**template,'allowed':True};queue.validate_permit(good,sha)
        for changed in ({'episodes':1200},{'host':'other'},{'queue_plan_sha256':'0'*64},{'evaluation_jobs':6},{'gpu_uuids':{'0':'x'}}):
            with self.assertRaises(ValueError):queue.validate_permit({**good,**changed},sha)
    def test_resource_requires_empty_70gib_reserved_card(self):
        good={'gpu':1,'uuid':queue.GPUS[1],'free_mib':81153,'used_mib':1,'utilization':0,'compute_pids':[]}
        self.assertTrue(queue.admissible(good))
        for change in ({'gpu':0},{'free_mib':71679},{'compute_pids':['45']},{'used_mib':65},{'utilization':1},{'uuid':'wrong'}):
            self.assertFalse(queue.admissible({**good,**change}))
    def test_single_use_consumption_write_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'AUTHORIZATION-CONSUMED.json';queue.write(p,{'bound':'example'},True)
            with self.assertRaises(FileExistsError):queue.write(p,{'bound':'example'},True)
    def test_200_300_denominators_and_identical_reset_bank(self):
        totals={}
        for job in self.plan['evaluations']:
            totals[job['needs_build']]=totals.get(job['needs_build'],0)+job['episodes']
            self.assertEqual(job['repeat'],'repeat-01');self.assertEqual(job['seed'],274001)
            self.assertEqual(len(job['expected_tasks']),10)
            for item in job['expected_tasks'].values():self.assertEqual(item['state_indices'],list(range(10)));self.assertEqual(len(item['raw_state_sha256']),10)
        self.assertEqual(sorted(totals.values()),[200,300])
    def test_exact_bool_outcomes_and_raw_reset_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            job=copy.deepcopy(self.plan['evaluations'][0]);job['output']=temp;path=Path(temp)
            info={'per_task':[{'task_id':i,'task_group':job['suite'],'metrics':{'successes':[True]*10}} for i in range(10)],'overall':{'n_episodes':100,'pc_success':100}}
            queue.write(path/'eval_info.json',info)
            entry=next(x['sha256'] for x in self.plan['assets'] if x['path']==str(queue.VLA/'scripts/eval_pi05_policy_with_procedural_bank.py'))
            reset={'bank_manifest_sha256':self.plan['bank']['sha256'],'selection_sha256':self.plan['selection']['sha256'],'repeat_id':'repeat-01','eval_seed':274001,'tasks':job['expected_tasks'],'entrypoint_sha256':entry,'eval_info_sha256':queue.digest(path/'eval_info.json')}
            queue.write(path/'procedural_bank_receipt.json',reset);queue.write(path/'bounded_resources.json',{'evaluation_completed':True,'changes_policy_rng':False,'changes_weights':False})
            result=queue.verify_evaluation(job,'a'*64,self.plan);self.assertEqual(result['successes'],100)
            info['per_task'][0]['metrics']['successes'][0]=1;queue.write(path/'eval_info.json',info);reset['eval_info_sha256']=queue.digest(path/'eval_info.json');queue.write(path/'procedural_bank_receipt.json',reset)
            with self.assertRaisesRegex(ValueError,'boolean'):queue.verify_evaluation(job,'a'*64,self.plan)
            info['per_task'][0]['metrics']['successes'][0]=True;queue.write(path/'eval_info.json',info);reset['eval_info_sha256']=queue.digest(path/'eval_info.json');reset['eval_seed']=274002;queue.write(path/'procedural_bank_receipt.json',reset)
            with self.assertRaisesRegex(ValueError,'Reset'):queue.verify_evaluation(job,'a'*64,self.plan)

    def test_transition_waits_for_reaped_child_context_without_signals(self):
        times=[0.];sleeps=[]
        busy={'gpu':1,'uuid':queue.GPUS[1],'free_mib':65000,'used_mib':16000,'utilization':10,'compute_pids':['42']}
        quiet={**busy,'compute_pids':[],'used_mib':128,'free_mib':81000,'utilization':0}
        idle={**quiet,'used_mib':1,'free_mib':81153}
        sequence=iter([busy,quiet,idle])
        def sleep(value):sleeps.append(value);times[0]+=value
        result=queue.wait_for_idle_transition(1,42,board_fn=lambda gpu:next(sequence),sleep_fn=sleep,clock=lambda:times[0],process_exists=lambda pid:False)
        self.assertEqual(sleeps,[2.,2.]);self.assertEqual(result['waited_seconds'],4.)
        self.assertTrue(result['same_leases_retained']);self.assertEqual(result['signals_sent'],0)
    def test_transition_foreign_pid_and_reused_pid_fail_closed(self):
        row={'gpu':1,'uuid':queue.GPUS[1],'free_mib':65000,'used_mib':16000,'utilization':10,'compute_pids':['99']}
        with self.assertRaisesRegex(ValueError,'Foreign'):
            queue.wait_for_idle_transition(1,42,board_fn=lambda gpu:row,sleep_fn=lambda s:self.fail('Must not wait on foreign owner'),process_exists=lambda pid:False)
        row['compute_pids']=['42']
        with self.assertRaisesRegex(ValueError,'Foreign'):
            queue.wait_for_idle_transition(1,42,board_fn=lambda gpu:row,process_exists=lambda pid:True)
    def test_transition_uuid_change_fails_without_releasing_owner(self):
        row={'gpu':1,'uuid':'wrong','compute_pids':[]}
        with self.assertRaisesRegex(ValueError,'UUID'):
            queue.wait_for_idle_transition(1,42,board_fn=lambda gpu:row)
    def test_transition_timeout_is_bounded(self):
        times=[0.];sleeps=[]
        row={'gpu':1,'uuid':queue.GPUS[1],'free_mib':80000,'used_mib':128,'utilization':0,'compute_pids':[]}
        def sleep(value):sleeps.append(value);times[0]+=value
        with self.assertRaises(TimeoutError):
            queue.wait_for_idle_transition(1,42,timeout_seconds=6.,poll_seconds=2.,board_fn=lambda gpu:row,sleep_fn=sleep,clock=lambda:times[0],process_exists=lambda pid:False)
        self.assertEqual(times[0],6.);self.assertEqual(sleeps,[2.,2.,2.])

if __name__=='__main__':unittest.main(verbosity=2)
