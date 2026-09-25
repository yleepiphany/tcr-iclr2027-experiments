import copy
import unittest
from prepare_readiness import select_observations, noise_id, GROUPS

def fixture():
    tasks=[];formal={}
    for gidx,group in enumerate(GROUPS):
        formal[group]=[]
        for task in range(10):
            identifier=gidx*10+task;name=f'task-{identifier}'
            formal[group].append({'task_index':identifier,'task':name})
            entry={'converted_episode':1000+identifier,'hash_rank':45,'frames':105,'split_hash':'h','source_hdf5_sha256':'s','trajectory_signature_sha256':'t'}
            tasks.append({'group_id':group,'task_index':identifier,'task':name,
                'splits':{'calibration':[entry],'train':[{'converted_episode':identifier}],'development':[{'converted_episode':500+identifier}]}})
    return {'tasks':tasks},formal

class Tests(unittest.TestCase):
    def test_fixed_150_observations_and_450_unique_noises(self):
        split,formal=fixture();obs=select_observations(split,formal)
        self.assertEqual(len(obs),150)
        seeds={noise_id(r['group'],r['task_slot'],r['request'],n) for r in obs for n in range(3)}
        self.assertEqual(len(seeds),450)
        self.assertEqual([r['frame_index'] for r in obs[:5]],[0,13,27,41,55])
    def test_target_labels_do_not_affect_selection(self):
        split,formal=fixture();expected=select_observations(split,formal)
        for r in split['tasks']:r.update(action='not-read',success=True)
        self.assertEqual(expected,select_observations(split,formal))
    def test_overlap_and_short_episodes_are_rejected(self):
        split,formal=fixture();split['tasks'][0]['splits']['train']=[{'converted_episode':1000}]
        with self.assertRaises(ValueError):select_observations(split,formal)
        split,formal=fixture();split['tasks'][0]['splits']['calibration'][0]['frames']=53
        with self.assertRaises(ValueError):select_observations(split,formal)

if __name__=='__main__':unittest.main()
