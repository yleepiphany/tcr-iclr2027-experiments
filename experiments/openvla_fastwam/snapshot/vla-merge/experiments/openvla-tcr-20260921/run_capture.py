"""New expert-only A/B capture on local GPU2/3; never resumes a failed batch."""
import argparse
import json
from pathlib import Path
import socket
import torch

import run_local as base
from run_dynamic import environment, UUIDS
from collect_experts import POOLS, verify_output
from evaluate import HORIZONS, load_selection, sha256_state

formal = base.formal
HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
ROOT = WORK / 'vla-merge-runtime/experiments/openvla-tcr-20260921'
OLD = ROOT / 'local-expert-dynamic-01'
SUITES = {'spatial': 'libero_spatial', 'object': 'libero_object', 'goal': 'libero_goal', 'long': 'libero_10'}


def verify(job):
    base.assert_unchanged(json.loads(Path(job['identity']).read_text())['files'])
    return verify_output(Path(job['output']) / 'capture', job)


def prepare(run):
    if run.exists():
        raise FileExistsError(run)
    if json.loads((OLD / 'queue-ended.json').read_text())['status'] != 'complete':
        raise ValueError('Native expert validation/evaluation did not finish')
    identity = json.loads((OLD / 'identities.json').read_text())
    base.assert_unchanged(identity['files'])
    selections = [load_selection(formal.BANK, formal.BANK / f'selections/repeat-0{i}.json', verify_source_files=True)
                  for i in (1, 2, 3)]
    from libero.libero import benchmark
    state_bank, tasks = {}, {}
    for expert, suite_name in SUITES.items():
        suite = benchmark.get_benchmark_dict()[suite_name]()
        if suite.n_tasks != 10:
            raise ValueError('Wrong task count')
        tasks[suite_name] = {}
        for task_id in range(10):
            task = suite.get_task(task_id)
            states = suite.get_task_init_states(task_id)
            formal_hashes = set()
            for selection in selections:
                selected = selection.tasks[(suite_name, task_id)]
                if (task.name, task.problem_folder, task.bddl_file) != (
                        selected.task_name, selected.problem_folder, selected.bddl_file):
                    raise ValueError('Task identity mismatch across calibration/evaluation banks')
                formal_hashes.update(selected.raw_sha256)
            pools = {}
            for pool, config in POOLS.items():
                if len(states) <= config['offset']:
                    raise ValueError('Not enough stock initial states; no wraparound')
                state = torch.as_tensor(states[config['offset']]).cpu().clone()
                digest = sha256_state(state.numpy())
                if digest in formal_hashes:
                    raise ValueError('Calibration reset content overlaps formal evaluation')
                state_bank[f'{pool}/{suite_name}/{task_id}'] = state
                pools[pool] = {'stock_offset': config['offset'], 'reset_sha256': digest}
            if pools['A']['reset_sha256'] == pools['B']['reset_sha256']:
                raise ValueError('A/B reset contents coincide')
            tasks[suite_name][str(task_id)] = {'name': task.name, 'problem_folder': task.problem_folder,
                'bddl_file': task.bddl_file, 'instruction': task.language, 'pools': pools,
                'excluded_formal_reset_hashes': sorted(formal_hashes)}
    run.mkdir(parents=True)
    states_path = run / 'init-states.pt'
    torch.save(state_bank, str(states_path))
    contract_path = run / 'capture-contract.json'
    formal.save(contract_path, {'schema': 'oft_expert_ab_capture_v1', 'pools': POOLS, 'tasks': tasks,
                'requests_per_episode': 5, 'selection': 'floor(j*(n_requests-1)/4), j=0..4',
                'insufficient_requests': 'fail, do not duplicate or choose another episode',
                'success_filter': False, 'native_chunk': 8, 'execute_actions': 8, 'warmup_steps': 10,
                'horizons': HORIZONS, 'calibration_episodes': 80, 'requests': 400,
                'states_sha256': formal.sha(states_path), 'formal_selection_sha256': [s.selection_sha256 for s in selections],
                'not_an_evaluation': True, 'not_a_block_validation_retry': True})
    for path in (states_path, contract_path, Path(__file__).resolve(), HERE / 'collect_experts.py', HERE / 'run_dynamic.py'):
        identity['files'][str(path)] = base.file_identity(path)
    formal.save(run / 'identities.json', identity)
    old = json.loads((OLD / 'smoke/plan.json').read_text())
    jobs = []
    for pool in POOLS:
        for expert, suite in SUITES.items():
            model = str(Path(identity['experts'][expert]['local_path']).resolve())
            output = run / 'jobs' / f'{pool}-{expert}'
            command = [str(base.PYTHON), str(HERE / 'collect_experts.py'), '--checkpoint', model,
                       '--contract', str(contract_path), '--states', str(states_path), '--output', str(output / 'capture'),
                       '--pool', pool, '--suite', suite]
            jobs.append({'id': output.name, 'output': str(output), 'identity': str(run / 'identities.json'),
                         'checkpoint': model, 'capture_contract': str(contract_path), 'pool': pool,
                         'suite': suite, 'expert': expert, 'command': command})
    plan = {**old, 'schema': 'oft_expert_ab_capture_queue_v1', 'run': str(run), 'jobs': jobs,
            'gpus': [2, 3], 'gpu_uuids': UUIDS, 'max_workers': 2, 'episodes': 0,
            'calibration_episodes': 80, 'requests': 400, 'min_free_mib': 32768, 'runtime_floor_mib': 12288,
            'identity_audit': str(run / 'identities.json'), 'identity_audit_sha256': formal.sha(run / 'identities.json'),
            'evaluator_sha256': formal.sha(HERE / 'collect_experts.py'),
            'environment_contract': {**old['environment_contract'], 'CUDA_VISIBLE_DEVICES': 'selected GPU 2 or 3'}}
    formal.save(run / 'plan.json', plan)
    formal.save(run / 'CLAIM.json', {'owner': 'Codex', 'host': socket.gethostname(), 'gpus': [2, 3],
                'scope': 'Expert-only A/B requests for OFT; no model regression/export or success evaluation',
                'no_retry': True, 'not_a_retry_of': 'block-interface-smoke-attempt-01'})
    print(json.dumps({'prepared': True, 'jobs': len(jobs), 'calibration_episodes': 80, 'requests': 400}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    args = p.parse_args()
    run = args.run.resolve()
    if socket.gethostname() != base.HOST or any(formal.gpu_row(g)['uuid'] != u for g, u in UUIDS.items()):
        raise ValueError('Wrong local host/card identity')
    if args.prepare:
        prepare(run)
        return
    base.assert_unchanged(json.loads((run / 'identities.json').read_text())['files'])
    formal.GPUS = (2, 3)
    formal.FLOOR_MIB = 12288
    formal.EVALUATOR = HERE / 'collect_experts.py'
    formal.IDENTITIES = run / 'identities.json'
    formal.scientific_environment = environment
    formal.verify_job = verify
    formal.save = base.save_with_actual_counts
    formal.run_queue(json.loads((run / 'plan.json').read_text()))


if __name__ == '__main__':
    main()
