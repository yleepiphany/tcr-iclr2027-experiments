"""Expert-only OFT A/B input capture; independent of block calibration validation."""
import argparse
from collections import deque
import json
from pathlib import Path
import numpy as np
import torch

from evaluate import HORIZONS, load_selection, sha256_state
from materialize_soup import sha, write
from native_oft import NativePolicy, validate_actions

POOLS = {'A': {'offset': 30, 'seed': 314100}, 'B': {'offset': 31, 'seed': 314200}}


def request_indices(count, quota=5):
    if type(count) is not int or type(quota) is not int or quota < 2 or count < quota:
        raise ValueError('Insufficient unique requests; do not duplicate or select another episode')
    return [j * (count - 1) // (quota - 1) for j in range(quota)]


def capture_rollout(policy, env, task, state, seed, horizon):
    native = policy.native
    native.set_seed_everywhere(seed)
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(state)
    for _ in range(10):
        obs, _, _, _ = env.step(native.get_libero_dummy_action('openvla'))
    queue, records, success, ended = deque(), [], False, False
    for step in range(horizon):
        if not queue:
            observation, _ = native.prepare_observation(obs, policy.resize_size)
            actions, request = policy.request(observation, task.language, capture=True)
            actions = validate_actions(actions)
            records.append({'request_index': len(records), 'action_step': step,
                            'inputs': request, 'native_actions': actions.copy()})
            queue.extend(actions)
        action = native.process_action(queue.popleft().copy(), 'openvla')
        obs, _, done, _ = env.step(action.tolist())
        success = bool(env.check_success())
        if done or success:
            ended = True
            break
    return records, {'success_diagnostic_only': success, 'action_steps': step + 1,
                     'environment_ended': ended, 'requests': len(records)}


def verify_output(output, job):
    output = Path(output)
    result = json.loads((output / 'summary.json').read_text())
    contract = json.loads(Path(job['capture_contract']).read_text())
    if (result.get('complete') is not True or result['pool'] != job['pool']
            or result['suite'] != job['suite'] or result['calibration_episodes'] != 10
            or result['requests'] != 50 or result['success_evaluation'] is not False
            or result['contract_sha256'] != sha(job['capture_contract'])
            or result['checkpoint'] != job['checkpoint']):
        raise ValueError('Incomplete or mismatched capture result')
    rows = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
    if len(rows) != 10 or {row['task_id'] for row in rows} != set(range(10)):
        raise ValueError('Wrong capture task coverage')
    for row in rows:
        expected = contract['tasks'][job['suite']][str(row['task_id'])]
        if (row['reset_sha256'] != expected['pools'][job['pool']]['reset_sha256']
                or row['instruction'] != expected['instruction']
                or row['seed'] != POOLS[job['pool']]['seed'] + row['task_id']
                or row['selected_indices'] != request_indices(row['requests'])):
            raise ValueError('Capture provenance mismatch')
        payload_path = output / row['payload']
        if sha(payload_path) != row['payload_sha256']:
            raise ValueError('Captured payload hash mismatch')
        payload = torch.load(str(payload_path), map_location='cpu', weights_only=False)
        if (payload['reset_sha256'] != row['reset_sha256'] or payload['instruction'] != row['instruction']
                or [r['request_index'] for r in payload['records']] != row['selected_indices']):
            raise ValueError('Payload identity mismatch')
        for record in payload['records']:
            inputs = record['inputs']
            if (record['identity_max_abs'] != 0 or record['action_step'] != record['request_index'] * 8
                    or inputs['attention_mask'].shape[0] != 1 or not bool(inputs['attention_mask'].all())
                    or inputs['pixel_values'].shape != (1, 12, 224, 224)
                    or np.asarray(inputs['proprio']).shape != (8,)):
                raise ValueError('Native capture contract mismatch')
            for value in inputs.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError('Nonfinite request tensor')
            validate_actions(record['native_actions'])
    if sha(output / 'episodes.jsonl') != result['episodes_sha256']:
        raise ValueError('Episode receipts changed')
    return {'episodes': 0, 'calibration_episodes': 10, 'requests': 50,
            'success_evaluation': False, 'summary_sha256': sha(output / 'summary.json')}


def main():
    p = argparse.ArgumentParser()
    for name in ('checkpoint', 'contract', 'states', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--pool', choices=POOLS, required=True)
    p.add_argument('--suite', choices=HORIZONS, required=True)
    args = p.parse_args()
    contract = json.loads(args.contract.read_text())
    if sha(args.states) != contract['states_sha256']:
        raise ValueError('Frozen reset vectors changed')
    if contract['pools'] != POOLS or contract['requests_per_episode'] != 5:
        raise ValueError('Unexpected frozen sampling contract')
    state_bank = torch.load(str(args.states), map_location='cpu', weights_only=True)
    args.output.mkdir(parents=True, exist_ok=False)
    policy = NativePolicy(args.checkpoint, args.suite)
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if suite.n_tasks != 10:
        raise ValueError('Wrong task count')
    for task_id in range(10):
        task = suite.get_task(task_id)
        expected = contract['tasks'][args.suite][str(task_id)]
        if (task.name, task.problem_folder, task.bddl_file, task.language) != (
                expected['name'], expected['problem_folder'], expected['bddl_file'], expected['instruction']):
            raise ValueError('Native task metadata changed')
        state = state_bank[f'{args.pool}/{args.suite}/{task_id}'].numpy().copy()
        digest = sha256_state(state)
        if digest != expected['pools'][args.pool]['reset_sha256']:
            raise ValueError('Actual reset input changed')
        seed = POOLS[args.pool]['seed'] + task_id
        env, _ = policy.native.get_libero_env(task, 'openvla', resolution=256)
        try:
            records, diagnostic = capture_rollout(policy, env, task, state, seed, HORIZONS[args.suite])
        finally:
            env.close()
        selected = request_indices(len(records))
        chosen = [records[i] for i in selected]
        for record in chosen:
            replay = policy.replay(record['inputs'])
            error = float(np.max(np.abs(replay - record['native_actions'])))
            if error != 0:
                raise ValueError('Collected native request failed exact replay')
            record['identity_max_abs'] = error
        payload = {'pool': args.pool, 'suite': args.suite, 'task_id': task_id, 'instruction': task.language,
                   'reset_sha256': digest, 'seed': seed, 'records': chosen}
        path = args.output / ('task-%02d.pt' % task_id)
        torch.save(payload, str(path))
        row = {key: value for key, value in payload.items() if key != 'records'}
        row.update(diagnostic, selected_indices=selected, payload=path.name, payload_sha256=sha(path))
        with (args.output / 'episodes.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
            stream.flush()
        print(json.dumps({'completed_tasks': task_id + 1, 'pool': args.pool, 'suite': args.suite}), flush=True)
    write(args.output / 'summary.json', {'complete': True, 'pool': args.pool, 'suite': args.suite,
          'calibration_episodes': 10, 'requests': 50, 'success_evaluation': False,
          'checkpoint': str(args.checkpoint.resolve()), 'contract_sha256': sha(args.contract),
          'episodes_sha256': sha(args.output / 'episodes.jsonl')})


if __name__ == '__main__':
    main()
