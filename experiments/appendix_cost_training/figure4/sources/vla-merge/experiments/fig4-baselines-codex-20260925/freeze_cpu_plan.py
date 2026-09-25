#!/usr/bin/env python3
"""Freeze only Figure 4 CPU planning/reuse evidence. Never starts GPU work."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

WORK = Path('/mnt/workspace/Wilson/parameter-fusion')
REPO = WORK / 'vla-merge'
EXPS = WORK / 'vla-merge-runtime/experiments'
NAME = 'fig4-baselines-codex-20260925'
OUT = EXPS / NAME
STATIC = EXPS / 'static-observation-baselines-20260922'
STATIC_CODE = REPO / 'experiments/static-observation-baselines-20260922'
TABLE1 = EXPS / 'iclr2027-table1-20260910'
SUBSET_SOURCE = EXPS / 'claude-pi05-expert-count-20260923'
SUBSETS = {'spatial-goal': ['spatial', 'goal'],
           'spatial-object-goal': ['spatial', 'object', 'goal']}
SUITES = {'spatial': 'libero_spatial', 'object': 'libero_object',
          'goal': 'libero_goal', 'long': 'libero_10'}

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def binding(p, expected=None, hash_now=True):
    p = Path(p)
    require(p.is_file(), f'Missing source: {p}')
    observed = sha(p) if hash_now else None
    if expected is not None and observed is not None:
        require(observed == expected, f'Source SHA changed: {p}')
    return {'path': str(p), 'bytes': p.stat().st_size,
            'sha256': observed or expected,
            'verification': 'sha256_recomputed' if hash_now else 'size_checked_hash_inherited_from_frozen_source'}

def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')

def four_expert_reuse():
    bank_root = TABLE1 / 'reset-banks/libero-procedural-clean-v1'
    bank_path = bank_root / 'manifest.json'
    bank = read(bank_path)['tasks']
    bank_sha = sha(bank_path)
    entry = REPO / 'scripts/eval_pi05_policy_with_procedural_bank.py'
    entry_sha = sha(entry)
    models = {
        'ties': TABLE1 / 'libero/ta-ties-fastlane-v2/candidates/ties_density_0p3_alpha_0p9/pretrained_model',
        'regmeanpp': STATIC / 'regmeanpp_spectral_smoothing/attempt-01/checkpoint',
        'featcal': STATIC / 'featcal/checkpoint/pretrained_model',
    }
    expected_shas = {
        'ties': '9382c79c9a76dea4803d71484599ec16edb77ee3cf7ce3b861fb9f0296921847',
        'regmeanpp': 'fea99a8c09f8ca16bf8a2889bac659aba86bcbf831ef422e8c0a30115bccffb2',
        'featcal': 'b0da07eda2a9ab4485f555f694d31afe08c40a39d6e3bfd44e9b9d8d006b0114',
    }
    methods = {}
    for method, model in models.items():
        jobs, rates = [], []
        for repeat in range(1, 4):
            repeat_id = f'repeat-{repeat:02d}'
            selection_path = bank_root / 'selections' / f'{repeat_id}.json'
            selection, selection_sha = read(selection_path)['tasks'], sha(selection_path)
            repeat_successes = 0
            for suite in SUITES.values():
                if method == 'ties':
                    job = TABLE1 / 'libero/clean-formal-ta-ties-v1/ties_merging' / repeat_id / suite / 'attempt-01'
                    run = read(job / 'run_receipt.json')
                    require(run['status'] == 'completed', 'TIES run is incomplete')
                    require(run['checkpoint']['model_sha256'] == expected_shas[method], 'TIES model binding differs')
                    checkpoint_manifest = run['checkpoint']['manifest']
                    binding(checkpoint_manifest['path'], checkpoint_manifest['sha256'])
                    command = run['command']
                else:
                    directory = 'regmeanpp_spectral_smoothing' if method == 'regmeanpp' else 'featcal'
                    job = STATIC / directory / 'evaluation/formal' / repeat_id / suite
                    run, receipt = read(job / 'running.json'), read(job / 'receipt.json')
                    require(run['model_sha256'] == receipt['model_sha256'] == expected_shas[method], 'Static model binding differs')
                    command = run['command']
                info_path, reset_path = job / 'eval/eval_info.json', job / 'eval/procedural_bank_receipt.json'
                info, reset = read(info_path), read(reset_path)
                require(reset['selection_sha256'] == selection_sha and reset['bank_manifest_sha256'] == bank_sha, 'Reset source differs')
                require(reset['entrypoint_sha256'] == entry_sha and reset['eval_seed'] == 274000 + repeat, 'Native evaluator/seed differs')
                require(reset['eval_info_sha256'] == sha(info_path), 'Result file changed')
                require(reset['repeat_id'] == repeat_id, 'Repeat differs')
                require(f'--policy.path={model}' in command and '--policy.n_action_steps=10' in command, 'Policy/runtime command differs')
                for flag in ['--env.hard_reset=true', '--eval.batch_size=1', '--eval.n_episodes=10', '--policy.compile_model=false']:
                    require(flag in command, f'Native command differs: {flag}')
                keys = {f'{suite}/{i:02d}' for i in range(10)}
                require(set(reset['tasks']) == keys, 'Reset task coverage differs')
                for key in keys:
                    selected = selection[key]
                    actual = reset['tasks'][key]
                    require(actual['state_indices'] == selected and actual['raw_state_sha256'] == [bank[key]['states'][i]['raw_sha256'] for i in selected], 'Actual reset identity differs')
                per_task = info['per_task']
                require(len(per_task) == 10 and {x['task_id'] for x in per_task} == set(range(10)), 'Task denominator differs')
                outcomes = [b for t in per_task for b in t['metrics']['successes']]
                require(len(outcomes) == 100 and all(type(b) is bool for b in outcomes), 'Episode denominator differs')
                require(info['overall']['n_episodes'] == 100 and abs(info['overall']['pc_success'] - sum(outcomes)) < 1e-6, 'Overall success differs')
                if method != 'ties':
                    require(receipt['sha256'] == sha(info_path) and receipt['successes'] == sum(outcomes), 'Static receipt/result differs')
                repeat_successes += sum(outcomes)
                jobs.append({'repeat': repeat_id, 'suite': suite, 'episodes': 100, 'successes': sum(outcomes),
                             'model_sha256': expected_shas[method], 'result': binding(info_path),
                             'reset_receipt': binding(reset_path), 'selection': binding(selection_path),
                             'native_command': command})
            rates.append(repeat_successes / 4)
        methods[method] = {'checkpoint': str(model),
                           'checkpoint_file': binding(model / 'model.safetensors', expected_shas[method], hash_now=False),
                           'checkpoint_binding': 'All 12 original launch/run receipts bind this immutable checkpoint SHA; model bytes not rehashed by this CPU-only plan freeze.',
                           'episodes': 1200, 'jobs': jobs, 'repeat_rates_percent': rates,
                           'mean_percent': statistics.mean(rates), 'sample_std_percent': statistics.stdev(rates)}
    return {'schema': 'fig4_three_baseline_four_expert_reuse_v1', 'accepted': True,
            'audited_at_utc': datetime.now(timezone.utc).isoformat(),
            'source': 'Original complete Table 1 runs, not a new measurement',
            'bank': binding(bank_path), 'native_entrypoint': binding(entry),
            'total_jobs': 36, 'total_episodes': 3600, 'methods': methods}

def prepare():
    require(not OUT.exists(), 'Figure 4 runtime already exists; no overwrite or duplicate plan')
    reuse = four_expert_reuse()
    feat_plan_path, reg_plan_path = STATIC / 'featcal/plan.json', STATIC / 'regmeanpp_spectral_smoothing/attempt-01/plan.json'
    feat, reg = read(feat_plan_path), read(reg_plan_path)
    subset_plan_path = SUBSET_SOURCE / 'plan-v1/plan.json'
    subsets = {x['id']: x for x in read(subset_plan_path)['subsets']}
    soup_proposal = read(WORK / 'coordination/2026-09-23/pi05-subset-soup-codex-transfer-proposal.json')
    soup_jobs = next(v for v in soup_proposal.values() if isinstance(v, list) and v and isinstance(v[0], dict) and 'subset_id' in v[0])
    soups = {x['subset_id']: x for x in soup_jobs}
    inputs = {'featcal': {}, 'regmeanpp': {}, 'subset_soups': {}}
    for expert in ['spatial', 'object', 'goal']:
        f, r = feat['raw'][expert], reg['inputs'][expert]
        inputs['featcal'][expert] = {'observations': binding(f['tensor_file'], f['tensor_sha256']),
                                     'manifest': binding(f['manifest'], f['manifest_sha256']),
                                     'observations_per_expert': 50, 'native_noise_replicas': 3,
                                     'calls_per_expert': 150, 'timestep': 1.0,
                                     'preserve_original_expert_ids_and_noise_seeds': True}
        inputs['regmeanpp'][expert] = {**r, 'replay': binding(r['replay_path'], r['replay_sha256'], hash_now=False),
                                       'manifest': binding(r['manifest_path'], r['manifest_sha256']),
                                       'observations_per_expert': 50, 'flow_indices': [0], 'timestep': 1.0}
    for subset in SUBSETS:
        x = soups[subset]
        require(x['subset_experts'] == SUBSETS[subset], 'Soup expert set differs')
        inputs['subset_soups'][subset] = {'checkpoint': x['checkpoint'], 'model_sha256': x['model_sha256'],
                                         'source_proposal': binding(WORK / 'coordination/2026-09-23/pi05-subset-soup-codex-transfer-proposal.json'),
                                         'checkpoint_file': binding(Path(x['checkpoint']) / 'model.safetensors', x['model_sha256'], hash_now=False),
                                         'role': 'Existing subset-only arithmetic mean initialization for FeatCal; verify exact mean and all sidecars before use'}
    vector_root = TABLE1 / 'libero/ta-ties-fastlane-v2/cache-v2/task-vectors-v2'
    vector_manifest = read(vector_root / 'cache-manifest.json')
    inputs['ties_task_vectors'] = {'manifest': binding(vector_root / 'cache-manifest.json'),
                                 'vectors': {e: binding(vector_root / f'{e}.f32.bin', vector_manifest['vectors'][e]['sha256'], hash_now=False) for e in ['spatial', 'object', 'goal']}}
    teacher_index_path = STATIC / 'featcal/shard-index.json'
    teacher_cache = STATIC / 'featcal/cache'
    teacher_specs = [s for s in read(teacher_index_path)['shards'] if s['role'] == 'teacher' and s['expert'] in ['spatial', 'object', 'goal']]
    require(len(teacher_specs) == 1410, 'Teacher shard count differs')
    for spec in teacher_specs:
        m = read(teacher_cache / spec['manifest_file'])
        require((teacher_cache / spec['data_file']).is_file() and m['status'] == 'complete' and m['role'] == 'teacher', 'Missing teacher shard')
        require(m['source_checkpoint_sha256'] == feat['models']['experts'][spec['expert']]['model_sha256'], 'Teacher source differs')
        require(m['plan_sha256'] == sha(feat_plan_path), 'Teacher source plan differs')
    inputs['featcal_teacher_reuse_candidate'] = {'cache_root': str(teacher_cache), 'index': binding(teacher_index_path),
        'teacher_shards_present': len(teacher_specs), 'data_bytes': sum((teacher_cache / s['data_file']).stat().st_size for s in teacher_specs),
        'status': 'metadata_and_presence_checked_not_rebound',
        'required_before_reuse': 'Verify every data SHA/row-key/source identity under its original plan; add immutable cross-plan provenance adapter. Never reuse old student shards or rewrite old manifests.'}
    source_paths = [REPO / 'scripts/ta_ties_core.py', REPO / 'scripts/materialize_iclr2027_table1_ta_ties_fastlane.py',
        STATIC_CODE / 'regmeanpp_spectral_smoothing_v1/materialize_smoothed.py', STATIC_CODE / 'regmeanpp_spectral_smoothing_v1/spectral_smoothing.py',
        STATIC_CODE / 'regmeanpp_spectral_smoothing_v1/smoothing_adapter.py', STATIC_CODE / 'regmeanpp_spectral_smoothing_v1/smoothing_contract.py',
        STATIC_CODE / 'featcal/run_observation_only_featcal.py', STATIC_CODE / 'featcal/featcal_observation_only.py', STATIC_CODE / 'featcal/featcal_parallel_solve.py',
        REPO / 'scripts/run_pi05_featcal_execution_pilot_v2.py', REPO / 'scripts/run_pi05_featcal_formal_full.py',
        REPO / 'src/vla_merge/featcal_hybrid_solve.py', REPO / 'scripts/eval_tcr_10k_formal_bounded.py',
        REPO / 'scripts/eval_pi05_policy_with_procedural_bank.py', Path(__file__).resolve()]
    algorithms = {
        'ties': {'density': 0.3, 'global_alpha': 0.9, 'trim': 'exact_ieee_float32_radix_global_per_expert',
                 'sign_election': 'mass', 'disjoint_aggregation': 'mean', 'modified_tensor_count': 422,
                 'subset_rule': 'Use only selected original per-expert task vectors; recompute sign election/disjoint mean for that subset; do not reuse four-expert merged direction.'},
        'regmeanpp': {'recipe': reg['recipe'], 'spectral_smoothing': reg['spectral_smoothing'], 'solved_weights': 418,
                     'bias_policy': reg['bias_rule'], 'subset_rule': 'Selected expert mean; selected static flow0 sources; merged-prefix full block replay; exact same fixed smoothing rule.'},
        'featcal': {k: feat[k] for k in ['alpha', 'lambda', 'rho', 'covariance_eps', 'bias_policy', 'noise_seed_base', 'timestep', 'solved_weights', 'stages']}}
    gaps = {
        'ties': ['Isolated subset-only vector/export adapter is absent; old fastlane CLI hard-locks four experts and the old candidate registry.',
                 'Reuse generic ta_ties_core.ties_direction with selected memmaps and the original 422-coordinate-segment scope, then independently verify selected-expert provenance/export.'],
        'regmeanpp': ['Isolated subset materializer/contract adapter is absent: old materialize_smoothed checks ORIGINAL_EXPERT_ORDER; smoothing_contract hardcodes four source names, source counts and total rows.',
                     'Parameterize only expert list/source bindings/mean initialization and expected total rows in a new directory; keep the tested smoothing kernel and original Gram/replay equations unchanged.'],
        'featcal': ['Isolated subset wrapper/shard schema adapter is absent: EXPERT_ORDER, initial STUDENT Soup, teacher completion groups, 4-expert shard contracts and provenance are hardcoded.',
                    'Use verified existing subset Soup; selected 150-call input groups and exact original noise IDs; rebind only independently verified teacher shards. Recompute every student capture and all 47 sequential solves.',
                    'The old parallel helper imports a four-expert baseline in its workers; subset worker context must be explicit and separately reviewed.']}
    builds, evals = [], []
    selection = TABLE1 / 'reset-banks/libero-procedural-clean-v1/selections/repeat-01.json'
    for method in ['featcal', 'regmeanpp', 'ties']:
        for subset, experts in SUBSETS.items():
            job_id = f'{method}-{subset}'
            checkpoint = OUT / 'checkpoints' / job_id / 'pretrained_model'
            builds.append({'id': job_id, 'method': method, 'subset': subset, 'experts': experts,
                           'checkpoint': str(checkpoint), 'status': 'blocked_missing_subset_adapter',
                           'launch_command': None, 'port_gaps': gaps[method],
                           'calibration_rows': (1110300 if method == 'featcal' else 592100 if method == 'regmeanpp' else 0) * len(experts),
                           'budget_basis': 'Original per-expert quota retained; TIES has no calibration',
                           'subset_bank': binding(subsets[subset]['bank'], subsets[subset]['bank_sha256'])})
            for expert in experts:
                suite = SUITES[expert]
                output = OUT / 'formal' / job_id / 'repeat-01' / suite / 'eval'
                command = [str(WORK / 'pi05_lora_finetune_v2_20260826/.venv/bin/python'), '-u', str(REPO / 'scripts/eval_tcr_10k_formal_bounded.py'),
                    f'--procedural-bank={selection.parent.parent}', f'--procedural-selection={selection}', f'--output_dir={output}',
                    '--env.type=libero', f'--env.task={suite}', '--env.init_states=true', '--env.hard_reset=true',
                    '--eval.batch_size=1', '--eval.n_episodes=10', '--seed=274001', f'--policy.path={checkpoint}',
                    '--policy.device=cuda', '--policy.compile_model=false', '--policy.gradient_checkpointing=false', '--policy.n_action_steps=10']
                evals.append({'id': f'{job_id}-{suite}', 'needs_build': job_id, 'method': method, 'subset': subset,
                              'suite': suite, 'repeat': 'repeat-01', 'seed': 274001, 'episodes': 100,
                              'output': str(output), 'command_template': command,
                              'status': 'blocked_until_checkpoint_native_acceptance_and_resource_claim'})
    require(len(builds) == 6 and len(evals) == 15 and sum(x['episodes'] for x in evals) == 1500, 'Unexpected new workload')
    plan = {'schema': 'fig4_three_baseline_subset_cpu_plan_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'request': 'FIG4-BASELINES-215', 'status': 'cpu_plan_frozen_gpu_blocked_missing_ports',
        'runtime_root': str(OUT), 'source_root': str(REPO / 'experiments' / NAME),
        'ownership': 'Exclusive CPU plan prepared by root/gpu_audit delegation; no GPU manager or card reservation created.',
        'no_training': True, 'no_parameter_search': True, 'no_score_based_scheduling': True, 'gpu_launch_allowed_by_this_plan': False,
        'source_binding': {'main_table': binding(REPO / 'paper/iclr2027/tabs/main_baselines.tex'),
            'current_figure_endpoint_convention': binding(WORK / 'coordination/2026-09-25/figure4-main-table-alignment.md'),
            'original_request': binding(WORK / 'coordination/2026-09-25/figure4-baseline-expansion-215.md'),
            'featcal_plan': binding(feat_plan_path), 'regmeanpp_plan': binding(reg_plan_path),
            'subset_plan': binding(subset_plan_path), 'dense_bank': binding(TABLE1 / 'expert-dense-bank-peft-v2.json')},
        'algorithm_sources': [binding(p) for p in source_paths], 'algorithms': algorithms, 'inputs': inputs,
        'figure_protocol': {'two_and_three_experts': 'One fixed subset checkpoint, repeat-01 only, 200 or 300 native episodes; no across-build error bars.',
            'four_experts': 'Reuse current Table 1 exact checkpoint, all three 400-episode repeats, mean and sample standard deviation.',
            'disclosure': 'Mixed n=1 subset points and three-repeat main-table four-expert endpoints; checkpoint/construction and calibration budgets differ. Descriptive comparison, not an isolated causal effect of expert count.',
            'selection_exposure': 'Prior Table 1 outcomes were already known. Subset rules are fixed before new subset outcomes and will not be scanned or tuned against them.'},
        'new_builds': builds, 'new_evaluations': evals, 'new_episode_total': 1500,
        'reused_four_expert_points': 3, 'reused_existing_jobs': 36, 'reused_existing_episodes': 3600,
        'evaluation_selection': binding(selection), 'four_expert_reuse_receipt': str(OUT / 'four-expert-reuse.json'),
        'shortest_route': ['Reuse the three already-complete four-expert endpoints; do not rebuild or re-evaluate them.',
            'Prepare subset TIES on CPU from existing global task-vector cache in parallel with isolated RegMean++ and FeatCal subset port work.',
            'Reuse verified subset Soup initialization and expert-only FeatCal teacher factors; do not reuse four-expert student states.',
            'After each of six exports passes finite/identity/native-action acceptance, run only its fixed two or three suites; no score-based gate or parameter change.',
            'All GPU dispatch waits for an independently accepted execution plan and explicit owner/card allocation; existing queues retain their jobs.'],
        'resource_snapshot_not_reservation': {'preferred_host_port': 1016, 'candidate_gpus': [1,4,5,6],
            'require_fresh_process_training_and_dual_lease_check_before_launch': True,
            'do_not_launch_new_work_on_1023_owned_queues': True}}
    OUT.mkdir(parents=True)
    write(OUT / 'four-expert-reuse.json', reuse)
    plan['four_expert_reuse_sha256'] = sha(OUT / 'four-expert-reuse.json')
    write(OUT / 'plan.json', plan)
    write(OUT / 'PLAN-SHA256.json', {'plan_sha256': sha(OUT / 'plan.json')})
    return preflight()

def preflight():
    plan = read(OUT / 'plan.json')
    require(sha(OUT / 'plan.json') == read(OUT / 'PLAN-SHA256.json')['plan_sha256'], 'Plan changed')
    for row in list(plan['source_binding'].values()) + plan['algorithm_sources']:
        binding(row['path'], row['sha256'])
    require(sha(OUT / 'four-expert-reuse.json') == plan['four_expert_reuse_sha256'], 'Reuse receipt changed')
    reuse = read(OUT / 'four-expert-reuse.json')
    require(reuse['accepted'] and reuse['total_jobs'] == 36 and reuse['total_episodes'] == 3600, 'Four-expert reuse differs')
    require(len(plan['new_builds']) == 6 and len(plan['new_evaluations']) == 15, 'Workload differs')
    ids = [j['id'] for j in plan['new_evaluations']]
    require(len(ids) == len(set(ids)) and sum(j['episodes'] for j in plan['new_evaluations']) == 1500, 'Evaluation duplication/denominator')
    require(all(j['launch_command'] is None for j in plan['new_builds']) and not plan['gpu_launch_allowed_by_this_plan'], 'CPU plan must not claim runnable GPU builds')
    result = {'status': 'pass_cpu_plan', 'gpu_execution': 'blocked_missing_subset_adapters',
              'plan_sha256': sha(OUT / 'plan.json'), 'new_builds': 6, 'new_jobs': 15, 'new_episodes': 1500,
              'reused_four_expert_jobs': 36, 'reused_four_expert_episodes': 3600,
              'gpu_jobs_launched': 0, 'training_jobs_launched': 0}
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'preflight'])
    action = parser.parse_args().action
    print(json.dumps(prepare() if action == 'prepare' else preflight(), indent=2))
